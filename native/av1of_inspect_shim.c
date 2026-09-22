/*
 * av1of_inspect_shim.c
 *
 *   Created by Julien Zouein on 22/06/2026.
 *   Copyright © 2026 Sigmedia.tv. All rights reserved.
 *   Copyright © 2026 Julien Zouein (zoueinj@tcd.ie)
 * --------------------------------------------------------------------------
 *
 * In-memory AV1 block-metadata extraction shim around patched libdav1d.
 *
 * This moves the per-frame array construction out of the Python callback and
 * into C: the whole file is decoded in a single call (so Python can release the
 * GIL for the entire multi-threaded decode), and the dav1d inspection callback
 * — which fires on dav1d's worker threads — transforms each frame's 4x4
 * `refmvs_block` grid directly into the final output layout WITHOUT touching the
 * Python interpreter. This removes the GIL serialization that otherwise pins the
 * pipeline near single-thread speed.
 *
 * Output per frame (matches src/modules/dav1d_inspect.py / the AOM JSON):
 *   motion_vectors : int16 [blk_h][blk_w][4] = (mv0_x, mv0_y, mv1_x, mv1_y), 1/8-pel
 *   reference_map  : int16 [blk_h][blk_w][2] = (ref0, ref1)
 *   block_map      : uint8 [blk_h][blk_w]    = AOM BLOCK_* enum
 *
 * Linked directly against our patched libdav1d with an rpath relative to the
 * shim, so it binds to that specific library (and is not interposed by another
 * libdav1d in the process, e.g. the one OpenCV bundles). See setup.sh:
 *   macOS:  cc ... -o libav1of_inspect.dylib shim.c \
 *               dav1d/build/src/libdav1d.7.dylib -Wl,-rpath,@loader_path/dav1d/build/src
 *   Linux:  cc ... -o libav1of_inspect.so shim.c \
 *               -L dav1d/build/src -ldav1d -Wl,-rpath,'$ORIGIN/dav1d/build/src'
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- *
 * Portability: a minimal mutex + symbol-export shim so the same source builds
 * on POSIX (pthreads) and Windows (SRWLOCK / __declspec(dllexport)).
 * ------------------------------------------------------------------------- */
#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  typedef SRWLOCK av1of_mutex;
  #define av1of_mutex_init(m)    InitializeSRWLock(m)
  #define av1of_mutex_destroy(m) ((void) (m))
  #define av1of_mutex_lock(m)    AcquireSRWLockExclusive(m)
  #define av1of_mutex_unlock(m)  ReleaseSRWLockExclusive(m)
  #define AV1OF_EXPORT           __declspec(dllexport)
#else
  #include <pthread.h>
  typedef pthread_mutex_t av1of_mutex;
  #define av1of_mutex_init(m)    pthread_mutex_init((m), NULL)
  #define av1of_mutex_destroy(m) pthread_mutex_destroy(m)
  #define av1of_mutex_lock(m)    pthread_mutex_lock(m)
  #define av1of_mutex_unlock(m)  pthread_mutex_unlock(m)
  #define AV1OF_EXPORT           __attribute__((visibility("default")))
#endif

#include <dav1d/dav1d.h>

// dav1d `enum BlockSize` (index) -> AOM `BLOCK_*` enum value, matched by WxH
// name (BS_128x128=0..BS_4x4=21 -> BLOCK_4X4=0..BLOCK_64X16=21).
static const uint8_t BS_DAV1D_TO_AOM[22] = {
    15, 14, 13, 12, 11, 21, 10, 9, 8, 19, 20, 7, 6, 5, 17, 18, 4, 3, 2, 16, 1, 0,
};

// INVALID_MV component sentinel (refmvs.h INVALID_MV == 0x80008000).
#define INVALID_MV_COMPONENT ((int16_t) 0x8000)

typedef struct {
    unsigned decode_seq;
    unsigned frame_offset;
    int frame_type;
    int width, height;
    int blk_w, blk_h;
    int8_t refidx[7];
    unsigned refpoc[7];
    int16_t *motion_vectors; // blk_h * blk_w * 4
    int16_t *reference_map;  // blk_h * blk_w * 2
    uint8_t *block_map;      // blk_h * blk_w
} av1of_frame;

typedef struct {
    av1of_frame *frames;
    int n, cap;
    av1of_mutex lock;
    int alloc_error;
} av1of_handle;

// One packed refmvs_block is 12 bytes:
//   int16 mv0_y, mv0_x, mv1_y, mv1_x; int8 ref0, ref1; uint8 bs, mf
#define REFMVS_BLOCK_SZ 12

static void inspect_cb(void *cookie, const Dav1dInspectData *d) {
    av1of_handle *const h = cookie;
    const int W = d->blk_w, H = d->blk_h;
    const size_t n = (size_t) W * H;

    av1of_frame f;
    memset(&f, 0, sizeof(f));
    f.decode_seq = d->decode_seq;
    f.frame_offset = d->frame_offset;
    f.frame_type = d->frame_type;
    f.width = d->width;
    f.height = d->height;
    f.blk_w = W;
    f.blk_h = H;
    memcpy(f.refidx, d->refidx, sizeof(f.refidx));
    memcpy(f.refpoc, d->refpoc, sizeof(f.refpoc));

    f.motion_vectors = malloc(n * 4 * sizeof(int16_t));
    f.reference_map = malloc(n * 2 * sizeof(int16_t));
    f.block_map = malloc(n);
    if (!f.motion_vectors || !f.reference_map || !f.block_map) {
        free(f.motion_vectors);
        free(f.reference_map);
        free(f.block_map);
        av1of_mutex_lock(&h->lock);
        h->alloc_error = 1;
        av1of_mutex_unlock(&h->lock);
        return;
    }

    const uint8_t *const blocks = d->blocks; // NULL only if metadata is unavailable
    const ptrdiff_t stride = d->blk_stride;  // in 12-byte records

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const size_t i = (size_t) y * W + x;
            int16_t *const mv = &f.motion_vectors[i * 4];
            int16_t *const ref = &f.reference_map[i * 2];

            if (!blocks) {
                // Intra / key frame: no motion field. Mirror AOM's intra
                // convention (zero MVs, ref = [0 = INTRA, -1 = none]).
                mv[0] = mv[1] = mv[2] = mv[3] = 0;
                ref[0] = 0;
                ref[1] = -1;
                f.block_map[i] = 0;
                continue;
            }

            const uint8_t *const rec =
                blocks + ((size_t) y * stride + x) * REFMVS_BLOCK_SZ;
            int16_t mv0_y, mv0_x, mv1_y, mv1_x;
            memcpy(&mv0_y, rec + 0, 2);
            memcpy(&mv0_x, rec + 2, 2);
            memcpy(&mv1_y, rec + 4, 2);
            memcpy(&mv1_x, rec + 6, 2);

            // Normalise INVALID_MV (0x8000) components to 0, as AOM does.
            if (mv0_x == INVALID_MV_COMPONENT) mv0_x = 0;
            if (mv0_y == INVALID_MV_COMPONENT) mv0_y = 0;
            if (mv1_x == INVALID_MV_COMPONENT) mv1_x = 0;
            if (mv1_y == INVALID_MV_COMPONENT) mv1_y = 0;

            // Channel order [x, y] per MV to match the .flo / AOM convention.
            mv[0] = mv0_x;
            mv[1] = mv0_y;
            mv[2] = mv1_x;
            mv[3] = mv1_y;
            ref[0] = (int8_t) rec[8];
            ref[1] = (int8_t) rec[9];

            const uint8_t bs = rec[10];
            f.block_map[i] = BS_DAV1D_TO_AOM[bs < 22 ? bs : 0];
        }
    }

    av1of_mutex_lock(&h->lock);
    if (h->n == h->cap) {
        const int cap = h->cap ? h->cap * 2 : 64;
        av1of_frame *const grown =
            realloc(h->frames, (size_t) cap * sizeof(av1of_frame));
        if (!grown) {
            h->alloc_error = 1;
            av1of_mutex_unlock(&h->lock);
            free(f.motion_vectors);
            free(f.reference_map);
            free(f.block_map);
            return;
        }
        h->frames = grown;
        h->cap = cap;
    }
    h->frames[h->n++] = f;
    av1of_mutex_unlock(&h->lock);
}

static void drain(Dav1dContext *const c, Dav1dPicture *const pic) {
    for (;;) {
        memset(pic, 0, sizeof(*pic));
        const int r = dav1d_get_picture(c, pic);
        if (r < 0) break; // DAV1D_ERR(EAGAIN) or genuine error: stop draining
        dav1d_picture_unref(pic);
    }
}

static int cmp_decode_seq(const void *a, const void *b) {
    const unsigned sa = ((const av1of_frame *) a)->decode_seq;
    const unsigned sb = ((const av1of_frame *) b)->decode_seq;
    return (sa > sb) - (sa < sb);
}

// Decode an entire IVF/AV1 file in memory, transforming every frame's block
// metadata in C. Returns 0 on success (*out owns the results), <0 on error.
AV1OF_EXPORT int av1of_decode(const char *const path, const int n_threads,
                 av1of_handle **const out) {
    *out = NULL;
    av1of_handle *const h = calloc(1, sizeof(*h));
    if (!h) return -1;
    av1of_mutex_init(&h->lock);

    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = n_threads;
    s.inspect_cb = inspect_cb;
    s.inspect_cookie = h;

    Dav1dContext *c = NULL;
    if (dav1d_open(&c, &s) < 0) {
        av1of_mutex_destroy(&h->lock);
        free(h);
        return -2;
    }

    FILE *const fp = fopen(path, "rb");
    if (!fp) {
        dav1d_close(&c);
        av1of_mutex_destroy(&h->lock);
        free(h);
        return -3;
    }

    int rc = 0;
    uint8_t hdr[32];
    if (fread(hdr, 1, 32, fp) != 32 || memcmp(hdr, "DKIF", 4) != 0) {
        rc = -4;
        goto done;
    }

    Dav1dPicture pic;
    uint8_t pkt_hdr[12];
    while (fread(pkt_hdr, 1, 12, fp) == 12) {
        const uint32_t sz = (uint32_t) pkt_hdr[0] | ((uint32_t) pkt_hdr[1] << 8) |
                            ((uint32_t) pkt_hdr[2] << 16) |
                            ((uint32_t) pkt_hdr[3] << 24);
        if (sz == 0) break;

        Dav1dData data;
        uint8_t *const buf = dav1d_data_create(&data, sz);
        if (!buf) {
            rc = -5;
            goto done;
        }
        if (fread(buf, 1, sz, fp) != sz) {
            dav1d_data_unref(&data);
            rc = -6;
            goto done;
        }

        while (data.sz > 0) {
            const int r = dav1d_send_data(c, &data);
            if (r < 0 && r != DAV1D_ERR(EAGAIN)) {
                dav1d_data_unref(&data);
                rc = -7;
                goto done;
            }
            drain(c, &pic);
        }
    }
    drain(c, &pic); // flush buffered frames at end of stream

done:
    fclose(fp);
    dav1d_close(&c); // joins worker threads: all callbacks have fired

    if (rc == 0 && h->alloc_error) rc = -8;
    if (rc != 0) {
        // Free any partial results.
        for (int i = 0; i < h->n; i++) {
            free(h->frames[i].motion_vectors);
            free(h->frames[i].reference_map);
            free(h->frames[i].block_map);
        }
        free(h->frames);
        av1of_mutex_destroy(&h->lock);
        free(h);
        return rc;
    }

    qsort(h->frames, h->n, sizeof(av1of_frame), cmp_decode_seq);
    *out = h;
    return 0;
}

AV1OF_EXPORT int av1of_num_frames(const av1of_handle *const h) { return h->n; }

AV1OF_EXPORT const av1of_frame *av1of_get_frame(const av1of_handle *const h, const int i) {
    return (i >= 0 && i < h->n) ? &h->frames[i] : NULL;
}

AV1OF_EXPORT void av1of_free(av1of_handle *const h) {
    if (!h) return;
    for (int i = 0; i < h->n; i++) {
        free(h->frames[i].motion_vectors);
        free(h->frames[i].reference_map);
        free(h->frames[i].block_map);
    }
    free(h->frames);
    av1of_mutex_destroy(&h->lock);
    free(h);
}

/* ------------------------------------------------------------------------- *
 * Additive RGB API.  The original av1of_* ABI above deliberately remains
 * unchanged: motion-only users still take exactly the same decode path.
 * ------------------------------------------------------------------------- */

#define AV1OF_RGB_PROCESS_MOTION 1u
#define AV1OF_RGB_LINEAR         2u
#define AV1OF_RGB_NORMALIZE      4u
#define AV1OF_RGB_NAN_TO_NUM     8u

typedef struct {
    unsigned decode_seq;
    unsigned frame_offset;
    int frame_type;
    int source_width, source_height;
    int width, height;             /* RGB / dense-field output size */
    int blk_w, blk_h;
    int8_t refidx[7];
    unsigned refpoc[7];
    int bpc;
    int matrix;
    int full_range;
    int ref_distance[8];           /* unwrapped current POC - reference POC */
    int16_t *motion_vectors;
    int16_t *reference_map;
    uint8_t *block_map;
    uint8_t *rgb;
    float *motion_field;
} av1of_rgb_frame;

typedef struct av1of_order_event {
    unsigned seq;
    unsigned offset;
    struct av1of_order_event *next;
} av1of_order_event;

typedef struct {
    av1of_rgb_frame *frames;
    int n, cap;
    av1of_mutex lock;
    int alloc_error;
    unsigned step, max_frames, flags;
    int target_width, target_height;
    unsigned next_order_seq;
    int order_counts[128];
    av1of_order_event *order_events;
} av1of_rgb_handle;

static int cmp_rgb_decode_seq(const void *a, const void *b) {
    const unsigned sa = ((const av1of_rgb_frame *) a)->decode_seq;
    const unsigned sb = ((const av1of_rgb_frame *) b)->decode_seq;
    return (sa > sb) - (sa < sb);
}

static int rgb_should_keep(const av1of_rgb_handle *const h,
                           const unsigned seq) {
    if (seq % h->step) return 0;
    return !h->max_frames || seq / h->step < h->max_frames;
}

/* Must be called with h->lock held. */
static av1of_rgb_frame *rgb_find_frame(av1of_rgb_handle *const h,
                                      const unsigned seq) {
    for (int i = 0; i < h->n; i++)
        if (h->frames[i].decode_seq == seq) return &h->frames[i];
    return NULL;
}

/* Must be called with h->lock held. */
static av1of_rgb_frame *rgb_find_or_add_frame(av1of_rgb_handle *const h,
                                             const unsigned seq) {
    av1of_rgb_frame *f = rgb_find_frame(h, seq);
    if (f) return f;
    if (h->n == h->cap) {
        const int cap = h->cap ? h->cap * 2 : 64;
        av1of_rgb_frame *const grown =
            realloc(h->frames, (size_t) cap * sizeof(*h->frames));
        if (!grown) {
            h->alloc_error = 1;
            return NULL;
        }
        h->frames = grown;
        h->cap = cap;
    }
    f = &h->frames[h->n++];
    memset(f, 0, sizeof(*f));
    f->decode_seq = seq;
    return f;
}

/* Inspection callbacks may arrive out of order on frame worker threads.  Keep
 * only the small outstanding order events and consume them as soon as the next
 * decode sequence appears.  This makes temporal reference distances match the
 * Python order-hint unwrapping without retaining metadata for skipped frames. */
static void rgb_record_order_locked(av1of_rgb_handle *const h,
                                    const unsigned seq,
                                    const unsigned offset) {
    av1of_order_event *const ev = malloc(sizeof(*ev));
    if (!ev) {
        h->alloc_error = 1;
        return;
    }
    ev->seq = seq;
    ev->offset = offset;
    av1of_order_event **at = &h->order_events;
    while (*at && (*at)->seq < seq) at = &(*at)->next;
    ev->next = *at;
    *at = ev;

    while (h->order_events && h->order_events->seq == h->next_order_seq) {
        av1of_order_event *const cur = h->order_events;
        const unsigned hint = cur->offset & 127u;
        h->order_counts[hint]++;
        av1of_rgb_frame *const f = rgb_find_frame(h, cur->seq);
        if (f) {
            const int frame_number = (int) hint + 128 * h->order_counts[hint];
            const int ref0 = 128 * h->order_counts[0];
            f->ref_distance[0] = frame_number - ref0;
            for (int i = 0; i < 7; i++) {
                const unsigned rh = f->refpoc[i] & 127u;
                const int ref_number = (int) rh + 128 * h->order_counts[rh];
                f->ref_distance[i + 1] = frame_number - ref_number;
            }
        }
        h->order_events = cur->next;
        free(cur);
        h->next_order_seq++;
    }
}

static void inspect_rgb_cb(void *cookie, const Dav1dInspectData *d) {
    av1of_rgb_handle *const h = cookie;
    const int keep = rgb_should_keep(h, d->decode_seq);
    const int W = d->blk_w, H = d->blk_h;
    const size_t n = (size_t) W * H;
    int16_t *mv_all = NULL, *ref_all = NULL;
    uint8_t *bs_all = NULL;

    if (keep) {
        mv_all = malloc(n * 4 * sizeof(*mv_all));
        ref_all = malloc(n * 2 * sizeof(*ref_all));
        bs_all = malloc(n);
        if (!mv_all || !ref_all || !bs_all) {
            free(mv_all);
            free(ref_all);
            free(bs_all);
            av1of_mutex_lock(&h->lock);
            h->alloc_error = 1;
            rgb_record_order_locked(h, d->decode_seq, d->frame_offset);
            av1of_mutex_unlock(&h->lock);
            return;
        }

        const uint8_t *const blocks = d->blocks;
        const ptrdiff_t stride = d->blk_stride;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                const size_t i = (size_t) y * W + x;
                int16_t *const mv = &mv_all[i * 4];
                int16_t *const ref = &ref_all[i * 2];
                if (!blocks) {
                    mv[0] = mv[1] = mv[2] = mv[3] = 0;
                    ref[0] = 0;
                    ref[1] = -1;
                    bs_all[i] = 0;
                    continue;
                }
                const uint8_t *const rec =
                    blocks + ((size_t) y * stride + x) * REFMVS_BLOCK_SZ;
                int16_t mv0_y, mv0_x, mv1_y, mv1_x;
                memcpy(&mv0_y, rec + 0, 2);
                memcpy(&mv0_x, rec + 2, 2);
                memcpy(&mv1_y, rec + 4, 2);
                memcpy(&mv1_x, rec + 6, 2);
                if (mv0_x == INVALID_MV_COMPONENT) mv0_x = 0;
                if (mv0_y == INVALID_MV_COMPONENT) mv0_y = 0;
                if (mv1_x == INVALID_MV_COMPONENT) mv1_x = 0;
                if (mv1_y == INVALID_MV_COMPONENT) mv1_y = 0;
                mv[0] = mv0_x;
                mv[1] = mv0_y;
                mv[2] = mv1_x;
                mv[3] = mv1_y;
                ref[0] = (int8_t) rec[8];
                ref[1] = (int8_t) rec[9];
                const uint8_t bs = rec[10];
                bs_all[i] = BS_DAV1D_TO_AOM[bs < 22 ? bs : 0];
            }
        }
    }

    av1of_mutex_lock(&h->lock);
    if (keep) {
        av1of_rgb_frame *const f = rgb_find_or_add_frame(h, d->decode_seq);
        if (f) {
            f->frame_offset = d->frame_offset;
            f->frame_type = d->frame_type;
            f->source_width = d->width;
            f->source_height = d->height;
            f->blk_w = W;
            f->blk_h = H;
            memcpy(f->refidx, d->refidx, sizeof(f->refidx));
            memcpy(f->refpoc, d->refpoc, sizeof(f->refpoc));
            f->motion_vectors = mv_all;
            f->reference_map = ref_all;
            f->block_map = bs_all;
            mv_all = ref_all = NULL;
            bs_all = NULL;
        }
    }
    rgb_record_order_locked(h, d->decode_seq, d->frame_offset);
    av1of_mutex_unlock(&h->lock);
    free(mv_all);
    free(ref_all);
    free(bs_all);
}

static uint8_t rgb_clip_q20(const int64_t value) {
    int64_t v = (value + (1 << 19)) >> 20;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t) v;
}

static int64_t rgb_coeff_q20(const double v) {
    return (int64_t) (v * (double) (1 << 20) + (v >= 0 ? 0.5 : -0.5));
}

static unsigned rgb_read_sample(const Dav1dPicture *const pic, const int plane,
                                const int x, const int y) {
    const uint8_t *const row = (const uint8_t *) pic->data[plane] +
                               (size_t) y * pic->stride[plane ? 1 : 0];
    if (pic->p.bpc <= 8) return row[x];
    uint16_t sample;
    memcpy(&sample, row + (size_t) x * 2, sizeof(sample));
    return sample;
}

static uint8_t *rgb_convert_picture(const Dav1dPicture *const pic,
                                    const int out_w, const int out_h,
                                    int *const matrix_out,
                                    int *const full_range_out) {
    const int w = pic->p.w, h = pic->p.h, bpc = pic->p.bpc;
    int matrix = DAV1D_MC_UNKNOWN;
    int full_range = 0;
    if (pic->seq_hdr) {
        matrix = pic->seq_hdr->mtrx;
        full_range = !!pic->seq_hdr->color_range;
    }
    *matrix_out = matrix;
    *full_range_out = full_range;

    if (out_w <= 0 || out_h <= 0) return NULL;
    if ((size_t) out_w > SIZE_MAX / 3 / (size_t) out_h) return NULL;
    uint8_t *const rgb = malloc((size_t) out_w * out_h * 3);
    if (!rgb) return NULL;
    const int maxv = (1 << bpc) - 1;
    const int black = full_range ? 0 : 16 << (bpc - 8);
    const int luma_span = full_range ? maxv : 219 << (bpc - 8);
    const int mid = 1 << (bpc - 1);
    const int chroma_span = full_range ? maxv : 224 << (bpc - 8);
    const int64_t yc = rgb_coeff_q20(255.0 / luma_span);

    double kr = 0.299, kb = 0.114;
    if (matrix == DAV1D_MC_BT709 ||
        (matrix != DAV1D_MC_IDENTITY && matrix != DAV1D_MC_BT470BG &&
         matrix != DAV1D_MC_BT601 && matrix != DAV1D_MC_BT2020_NCL &&
         matrix != DAV1D_MC_BT2020_CL && h >= 720)) {
        kr = 0.2126; kb = 0.0722;
    } else if (matrix == DAV1D_MC_BT2020_NCL || matrix == DAV1D_MC_BT2020_CL) {
        kr = 0.2627; kb = 0.0593;
    }
    const double kg = 1.0 - kr - kb;
    const int64_t rc = rgb_coeff_q20(255.0 * (2.0 - 2.0 * kr) / chroma_span);
    const int64_t bc = rgb_coeff_q20(255.0 * (2.0 - 2.0 * kb) / chroma_span);
    const int64_t guc = rgb_coeff_q20(
        -255.0 * (2.0 * kb * (1.0 - kb) / kg) / chroma_span);
    const int64_t gvc = rgb_coeff_q20(
        -255.0 * (2.0 * kr * (1.0 - kr) / kg) / chroma_span);

    for (int oy = 0; oy < out_h; oy++) {
        const int sy = (int) ((int64_t) oy * h / out_h);
        for (int ox = 0; ox < out_w; ox++) {
            const int sx = (int) ((int64_t) ox * w / out_w);
            const unsigned Y = rgb_read_sample(pic, 0, sx, sy);
            uint8_t *const dst = &rgb[((size_t) oy * out_w + ox) * 3];
            if (pic->p.layout == DAV1D_PIXEL_LAYOUT_I400 ||
                !pic->data[1] || !pic->data[2]) {
                const uint8_t g = rgb_clip_q20(((int64_t) Y - black) * yc);
                dst[0] = dst[1] = dst[2] = g;
                continue;
            }
            const int cx = pic->p.layout == DAV1D_PIXEL_LAYOUT_I444 ? sx : sx / 2;
            const int cy = pic->p.layout == DAV1D_PIXEL_LAYOUT_I420 ? sy / 2 : sy;
            const unsigned U = rgb_read_sample(pic, 1, cx, cy);
            const unsigned V = rgb_read_sample(pic, 2, cx, cy);
            if (matrix == DAV1D_MC_IDENTITY) {
                dst[0] = rgb_clip_q20(((int64_t) V - black) * yc);
                dst[1] = rgb_clip_q20(((int64_t) Y - black) * yc);
                dst[2] = rgb_clip_q20(((int64_t) U - black) * yc);
                continue;
            }
            const int64_t yq = ((int64_t) Y - black) * yc;
            const int64_t uq = (int64_t) U - mid;
            const int64_t vq = (int64_t) V - mid;
            dst[0] = rgb_clip_q20(yq + vq * rc);
            dst[1] = rgb_clip_q20(yq + uq * guc + vq * gvc);
            dst[2] = rgb_clip_q20(yq + uq * bc);
        }
    }
    return rgb;
}

static void drain_rgb(Dav1dContext *const c, Dav1dPicture *const pic,
                      av1of_rgb_handle *const h) {
    for (;;) {
        memset(pic, 0, sizeof(*pic));
        const int r = dav1d_get_picture(c, pic);
        if (r < 0) break;
        const unsigned seq = (unsigned) pic->m.timestamp;
        if (rgb_should_keep(h, seq)) {
            const int out_w = h->target_width > 0 ? h->target_width : pic->p.w;
            const int out_h = h->target_height > 0 ? h->target_height : pic->p.h;
            int matrix, full_range;
            uint8_t *const rgb = rgb_convert_picture(pic, out_w, out_h,
                                                     &matrix, &full_range);
            av1of_mutex_lock(&h->lock);
            if (!rgb) {
                h->alloc_error = 1;
            } else {
                av1of_rgb_frame *const f = rgb_find_or_add_frame(h, seq);
                if (f) {
                    free(f->rgb);
                    f->rgb = rgb;
                    f->width = out_w;
                    f->height = out_h;
                    f->bpc = pic->p.bpc;
                    f->matrix = matrix;
                    f->full_range = full_range;
                } else {
                    free(rgb);
                }
            }
            av1of_mutex_unlock(&h->lock);
        }
        dav1d_picture_unref(pic);
    }
}

static void rgb_make_motion_fields(av1of_rgb_handle *const h) {
    if (!(h->flags & AV1OF_RGB_PROCESS_MOTION)) return;
    for (int i = 0; i < h->n; i++) {
        av1of_rgb_frame *const f = &h->frames[i];
        if (!f->motion_vectors || !f->reference_map ||
            f->width <= 0 || f->height <= 0 || f->blk_w <= 0 || f->blk_h <= 0)
            continue;
        const size_t count = (size_t) f->width * f->height;
        if (count > SIZE_MAX / 2 / sizeof(*f->motion_field)) {
            h->alloc_error = 1;
            return;
        }
        f->motion_field = malloc(count * 2 * sizeof(*f->motion_field));
        if (!f->motion_field) {
            h->alloc_error = 1;
            return;
        }
        for (int y = 0; y < f->height; y++) {
            const int sy = (int) ((int64_t) y * f->source_height / f->height);
            int by = sy / 4;
            if (by >= f->blk_h) by = f->blk_h - 1;
            for (int x = 0; x < f->width; x++) {
                const int sx = (int) ((int64_t) x * f->source_width / f->width);
                int bx = sx / 4;
                if (bx >= f->blk_w) bx = f->blk_w - 1;
                const size_t bi = (size_t) by * f->blk_w + bx;
                float fx = f->motion_vectors[bi * 4] / 8.0f;
                float fy = f->motion_vectors[bi * 4 + 1] / 8.0f;
                if (h->flags & AV1OF_RGB_LINEAR) {
                    const int ref = f->reference_map[bi * 2];
                    const int distance = ref >= 0 && ref < 8 ?
                                         f->ref_distance[ref] : 0;
                    if (distance) {
                        fx /= distance;
                        fy /= distance;
                    } else if (h->flags & AV1OF_RGB_NAN_TO_NUM) {
                        fx = fy = 0.0f;
                    } else {
                        fx = fy = NAN;
                    }
                }
                if (h->flags & AV1OF_RGB_NORMALIZE) {
                    fx /= f->source_width;
                    fy /= f->source_height;
                    if (fx < -1.0f) fx = -1.0f;
                    if (fx > 1.0f) fx = 1.0f;
                    if (fy < -1.0f) fy = -1.0f;
                    if (fy > 1.0f) fy = 1.0f;
                }
                f->motion_field[((size_t) y * f->width + x) * 2] = fx;
                f->motion_field[((size_t) y * f->width + x) * 2 + 1] = fy;
            }
        }
    }
}

static void rgb_free_payload(av1of_rgb_frame *const f) {
    free(f->motion_vectors); f->motion_vectors = NULL;
    free(f->reference_map); f->reference_map = NULL;
    free(f->block_map); f->block_map = NULL;
    free(f->rgb); f->rgb = NULL;
    free(f->motion_field); f->motion_field = NULL;
}

/* Decode sampled frames to RGB and, optionally, a dense backward field.  Flags
 * are AV1OF_RGB_* above. target_width/height must both be zero or both > zero;
 * step==0 is rejected and max_frames==0 means unlimited. */
AV1OF_EXPORT int av1of_decode_rgb(const char *const path, const int n_threads,
                 const int target_width, const int target_height,
                 const unsigned step, const unsigned max_frames,
                 const unsigned flags, av1of_rgb_handle **const out) {
    *out = NULL;
    if (!step || target_width < 0 || target_height < 0 ||
        (!!target_width != !!target_height)) return -9;
    av1of_rgb_handle *const h = calloc(1, sizeof(*h));
    if (!h) return -1;
    av1of_mutex_init(&h->lock);
    h->step = step;
    h->max_frames = max_frames;
    h->flags = flags;
    h->target_width = target_width;
    h->target_height = target_height;
    for (int i = 0; i < 128; i++) h->order_counts[i] = -1;

    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = n_threads;
    s.inspect_cb = inspect_rgb_cb;
    s.inspect_cookie = h;
    Dav1dContext *c = NULL;
    if (dav1d_open(&c, &s) < 0) {
        av1of_mutex_destroy(&h->lock);
        free(h);
        return -2;
    }
    FILE *const fp = fopen(path, "rb");
    if (!fp) {
        dav1d_close(&c);
        av1of_mutex_destroy(&h->lock);
        free(h);
        return -3;
    }

    int rc = 0;
    uint8_t hdr[32];
    if (fread(hdr, 1, 32, fp) != 32 || memcmp(hdr, "DKIF", 4) != 0) {
        rc = -4;
        goto rgb_done;
    }
    Dav1dPicture pic;
    uint8_t pkt_hdr[12];
    unsigned packet_seq = 0;
    while (fread(pkt_hdr, 1, 12, fp) == 12) {
        if (max_frames && (uint64_t) packet_seq >
                          (uint64_t) (max_frames - 1) * step) break;
        const uint32_t sz = (uint32_t) pkt_hdr[0] | ((uint32_t) pkt_hdr[1] << 8) |
                            ((uint32_t) pkt_hdr[2] << 16) |
                            ((uint32_t) pkt_hdr[3] << 24);
        if (!sz) break;
        Dav1dData data;
        uint8_t *const buf = dav1d_data_create(&data, sz);
        if (!buf) { rc = -5; goto rgb_done; }
        if (fread(buf, 1, sz, fp) != sz) {
            dav1d_data_unref(&data);
            rc = -6;
            goto rgb_done;
        }
        data.m.timestamp = packet_seq++;
        while (data.sz > 0) {
            const int r = dav1d_send_data(c, &data);
            if (r < 0 && r != DAV1D_ERR(EAGAIN)) {
                dav1d_data_unref(&data);
                rc = -7;
                goto rgb_done;
            }
            drain_rgb(c, &pic, h);
        }
    }
    drain_rgb(c, &pic, h);

rgb_done:
    fclose(fp);
    dav1d_close(&c);
    while (h->order_events) {
        av1of_order_event *const next = h->order_events->next;
        free(h->order_events);
        h->order_events = next;
    }
    if (!rc && h->alloc_error) rc = -8;
    if (rc) {
        for (int i = 0; i < h->n; i++) rgb_free_payload(&h->frames[i]);
        free(h->frames);
        av1of_mutex_destroy(&h->lock);
        free(h);
        return rc;
    }
    qsort(h->frames, h->n, sizeof(*h->frames), cmp_rgb_decode_seq);
    rgb_make_motion_fields(h);
    if (h->alloc_error) {
        for (int i = 0; i < h->n; i++) rgb_free_payload(&h->frames[i]);
        free(h->frames);
        av1of_mutex_destroy(&h->lock);
        free(h);
        return -8;
    }
    *out = h;
    return 0;
}

AV1OF_EXPORT int av1of_rgb_num_frames(const av1of_rgb_handle *const h) {
    return h ? h->n : 0;
}

AV1OF_EXPORT const av1of_rgb_frame *av1of_rgb_get_frame(
        const av1of_rgb_handle *const h, const int i) {
    return h && i >= 0 && i < h->n ? &h->frames[i] : NULL;
}

/* Call after copying frame i to release its potentially large buffers early. */
AV1OF_EXPORT void av1of_rgb_release_frame(av1of_rgb_handle *const h, const int i) {
    if (!h || i < 0 || i >= h->n) return;
    rgb_free_payload(&h->frames[i]);
}

AV1OF_EXPORT void av1of_rgb_free(av1of_rgb_handle *const h) {
    if (!h) return;
    for (int i = 0; i < h->n; i++) rgb_free_payload(&h->frames[i]);
    free(h->frames);
    av1of_mutex_destroy(&h->lock);
    free(h);
}
