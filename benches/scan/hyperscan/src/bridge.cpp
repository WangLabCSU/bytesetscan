#include "nfa/shufticompile.h"
#include "nfa/trufflecompile.h"
#include "util/simd_utils.h"

#include <cstddef>
#include <new>

// Upstream's fat runtime prefixes each architecture's internal symbols.
extern "C" {
const u8 *core2_shuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *core2_rshuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *core2_truffleExec(m128, m128, const u8 *, const u8 *);
const u8 *core2_rtruffleExec(m128, m128, const u8 *, const u8 *);
const u8 *corei7_shuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *corei7_rshuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *corei7_truffleExec(m128, m128, const u8 *, const u8 *);
const u8 *corei7_rtruffleExec(m128, m128, const u8 *, const u8 *);
const u8 *avx2_shuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *avx2_rshuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *avx2_truffleExec(m128, m128, const u8 *, const u8 *);
const u8 *avx2_rtruffleExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512_shuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512_rshuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512_truffleExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512_rtruffleExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512vbmi_shuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512vbmi_rshuftiExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512vbmi_truffleExec(m128, m128, const u8 *, const u8 *);
const u8 *avx512vbmi_rtruffleExec(m128, m128, const u8 *, const u8 *);
}

using Search = const u8 *(*)(m128, m128, const u8 *, const u8 *);

struct Matcher {
    u8 low[16];
    u8 high[16];
    Search find;
    Search rfind;
};

extern "C" Matcher *scan_new(unsigned algorithm, unsigned backend,
                              const u8 *needle, size_t len) noexcept {
    auto *matcher = new Matcher{};
    ue2::CharReach characters;
    for (size_t i = 0; i < len; ++i) {
        characters.set(needle[i]);
    }
    if (algorithm == 0) {
        if (ue2::shuftiBuildMasks(characters, matcher->low, matcher->high) < 0) {
            delete matcher;
            return nullptr;
        }
    } else {
        ue2::truffleBuildMasks(characters, matcher->low, matcher->high);
    }
    switch (backend) {
    case 0:
        matcher->find = algorithm == 0 ? core2_shuftiExec : core2_truffleExec;
        matcher->rfind = algorithm == 0 ? core2_rshuftiExec : core2_rtruffleExec;
        break;
    case 1:
        matcher->find = algorithm == 0 ? corei7_shuftiExec : corei7_truffleExec;
        matcher->rfind = algorithm == 0 ? corei7_rshuftiExec : corei7_rtruffleExec;
        break;
    case 2:
        matcher->find = algorithm == 0 ? avx2_shuftiExec : avx2_truffleExec;
        matcher->rfind = algorithm == 0 ? avx2_rshuftiExec : avx2_rtruffleExec;
        break;
    case 3:
        matcher->find = algorithm == 0 ? avx512_shuftiExec : avx512_truffleExec;
        matcher->rfind = algorithm == 0 ? avx512_rshuftiExec : avx512_rtruffleExec;
        break;
    case 4:
        matcher->find = algorithm == 0 ? avx512vbmi_shuftiExec : avx512vbmi_truffleExec;
        matcher->rfind = algorithm == 0 ? avx512vbmi_rshuftiExec : avx512vbmi_rtruffleExec;
        break;
    default:
        delete matcher;
        return nullptr;
    }
    return matcher;
}

extern "C" size_t scan_find(const Matcher *matcher, const u8 *buf, size_t len) {
    // Original kernels require nonempty input. Handle it before entering them.
    if (len == 0) {
        return len;
    }
    const u8 *result = matcher->find(loadu128(matcher->low),
                                    loadu128(matcher->high), buf, buf + len);
    return static_cast<size_t>(result - buf);
}

extern "C" size_t scan_rfind(const Matcher *matcher, const u8 *buf, size_t len) {
    if (len == 0) {
        return len;
    }
    const u8 *result = matcher->rfind(loadu128(matcher->low),
                                     loadu128(matcher->high), buf, buf + len);
    // The caller supplies prefix storage in the same allocation, so the
    // original kernel's buf - 1 sentinel is a valid pointer.
    return result == buf - 1 ? len : static_cast<size_t>(result - buf);
}

extern "C" void scan_free(Matcher *matcher) {
    delete matcher;
}
