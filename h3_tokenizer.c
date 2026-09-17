#include "h3_tokenizer.h"

#include "h3_json.h"

#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

typedef struct {
    const char *key;
    uint32_t value;
} token_map_slot;

typedef struct {
    token_map_slot *slots;
    size_t capacity;
    int owns_keys;
} token_map;

typedef struct {
    uint32_t value;
    size_t location;
    size_t length;
} codepoint;

typedef struct {
    char **items;
    size_t count;
} string_list;

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} id_list;

struct h3_tokenizer {
    h3_json_value *document;
    token_map vocab;
    token_map merge_ranks;
    token_map added_tokens;
    const char **inverse_vocab;
    const char **inverse_added;
    size_t inverse_count;
    const char **added_alternatives;
    size_t added_count;
    char *byte_encoder[256];
    int16_t byte_decoder[324];
};

static void tok_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static size_t string_hash(const char *key) {
    size_t hash = sizeof(size_t) >= 8 ?
        (size_t)UINT64_C(1469598103934665603) : (size_t)UINT32_C(2166136261);
    size_t prime = sizeof(size_t) >= 8 ?
        (size_t)UINT64_C(1099511628211) : (size_t)UINT32_C(16777619);
    for (const unsigned char *cursor = (const unsigned char *)key; *cursor;
         cursor++) {
        hash ^= *cursor;
        hash *= prime;
    }
    return hash;
}

static int map_init(token_map *map, size_t expected, int owns_keys) {
    size_t capacity = 16;
    if (expected > SIZE_MAX / 2) return 0;
    while (capacity < expected * 2) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity *= 2;
    }
    map->slots = calloc(capacity, sizeof(*map->slots));
    if (!map->slots) return 0;
    map->capacity = capacity;
    map->owns_keys = owns_keys;
    return 1;
}

static void map_free(token_map *map) {
    if (map->owns_keys)
        for (size_t index = 0; index < map->capacity; index++)
            free((char *)map->slots[index].key);
    free(map->slots);
    memset(map, 0, sizeof(*map));
}

static const token_map_slot *map_find(const token_map *map, const char *key) {
    if (!map->capacity) return NULL;
    size_t mask = map->capacity - 1;
    size_t slot = string_hash(key) & mask;
    while (map->slots[slot].key) {
        if (!strcmp(map->slots[slot].key, key)) return &map->slots[slot];
        slot = (slot + 1) & mask;
    }
    return NULL;
}

static int map_insert(token_map *map, const char *key, uint32_t value) {
    size_t mask = map->capacity - 1;
    size_t slot = string_hash(key) & mask;
    while (map->slots[slot].key) {
        if (!strcmp(map->slots[slot].key, key)) return 0;
        slot = (slot + 1) & mask;
    }
    map->slots[slot] = (token_map_slot){key, value};
    return 1;
}

static char *copy_range(const char *text, size_t start, size_t stop) {
    if (stop < start || stop - start == SIZE_MAX) return NULL;
    char *copy = malloc(stop - start + 1);
    if (!copy) return NULL;
    memcpy(copy, text + start, stop - start);
    copy[stop - start] = '\0';
    return copy;
}

static int strings_push(string_list *list, char *item) {
    if (list->count == SIZE_MAX / sizeof(*list->items)) return 0;
    char **items = realloc(list->items, (list->count + 1) * sizeof(*items));
    if (!items) return 0;
    list->items = items;
    list->items[list->count++] = item;
    return 1;
}

static void strings_free(string_list *list) {
    for (size_t index = 0; index < list->count; index++) free(list->items[index]);
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int ids_push(id_list *list, uint32_t value) {
    if (list->count == list->capacity) {
        size_t capacity = list->capacity ? list->capacity * 2 : 16;
        if (capacity < list->capacity ||
            capacity > SIZE_MAX / sizeof(*list->items)) return 0;
        uint32_t *items = realloc(list->items, capacity * sizeof(*items));
        if (!items) return 0;
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = value;
    return 1;
}

static char *normalize_nfc(const char *utf8, char *error, size_t error_size) {
    if (strlen(utf8) > INT32_MAX) {
        tok_fail(error, error_size, "prompt is too large");
        return NULL;
    }
    UErrorCode status = U_ZERO_ERROR;
    int32_t source_units = 0;
    u_strFromUTF8(NULL, 0, &source_units, utf8, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        tok_fail(error, error_size, "prompt is not valid UTF-8");
        return NULL;
    }
    status = U_ZERO_ERROR;
    UChar *source = malloc(((size_t)source_units + 1) * sizeof(*source));
    if (!source) goto memory_failure;
    u_strFromUTF8(source, source_units + 1, NULL, utf8, -1, &status);
    if (U_FAILURE(status)) {
        free(source);
        tok_fail(error, error_size, "prompt is not valid UTF-8");
        return NULL;
    }
    status = U_ZERO_ERROR;
    const UNormalizer2 *normalizer = unorm2_getNFCInstance(&status);
    int32_t normalized_units = unorm2_normalize(
        normalizer, source, source_units, NULL, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(source);
        tok_fail(error, error_size, "cannot normalize prompt as NFC");
        return NULL;
    }
    status = U_ZERO_ERROR;
    UChar *normalized = malloc(
        ((size_t)normalized_units + 1) * sizeof(*normalized));
    if (!normalized) {
        free(source);
        goto memory_failure;
    }
    unorm2_normalize(normalizer, source, source_units, normalized,
                     normalized_units + 1, &status);
    free(source);
    if (U_FAILURE(status)) {
        free(normalized);
        tok_fail(error, error_size, "cannot normalize prompt as NFC");
        return NULL;
    }
    status = U_ZERO_ERROR;
    int32_t output_bytes = 0;
    u_strToUTF8(NULL, 0, &output_bytes, normalized, normalized_units, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
        free(normalized);
        tok_fail(error, error_size, "cannot encode normalized prompt");
        return NULL;
    }
    status = U_ZERO_ERROR;
    char *output = malloc((size_t)output_bytes + 1);
    if (!output) {
        free(normalized);
        goto memory_failure;
    }
    u_strToUTF8(output, output_bytes + 1, NULL, normalized, normalized_units,
                &status);
    free(normalized);
    if (U_FAILURE(status)) {
        free(output);
        tok_fail(error, error_size, "cannot encode normalized prompt");
        return NULL;
    }
    output[output_bytes] = '\0';
    return output;

memory_failure:
    tok_fail(error, error_size, "out of memory normalizing prompt");
    return NULL;
}

static codepoint *codepoints(const char *text, size_t *count,
                             char *error, size_t error_size) {
    size_t length = strlen(text);
    if (length > INT32_MAX || length == SIZE_MAX / sizeof(codepoint)) {
        tok_fail(error, error_size, "prompt is too large");
        return NULL;
    }
    codepoint *points = malloc((length + 1) * sizeof(*points));
    if (!points) {
        tok_fail(error, error_size, "out of memory reading prompt");
        return NULL;
    }
    int32_t offset = 0;
    size_t used = 0;
    while (offset < (int32_t)length) {
        int32_t start = offset;
        UChar32 value;
        U8_NEXT(text, offset, (int32_t)length, value);
        if (value < 0) {
            free(points);
            tok_fail(error, error_size, "prompt is not valid UTF-8");
            return NULL;
        }
        points[used++] = (codepoint){
            (uint32_t)value, (size_t)start, (size_t)(offset - start)
        };
    }
    *count = used;
    return points;
}

static int is_letter(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
           category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
           category == U_OTHER_LETTER;
}

static int is_number(uint32_t value) {
    int8_t category = u_charType((UChar32)value);
    return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
           category == U_OTHER_NUMBER;
}

static int is_space(uint32_t value) {
    return u_isUWhiteSpace((UChar32)value) ||
           (value >= 0x1c && value <= 0x1f);
}

static char *slice_points(const char *text, const codepoint *points,
                          size_t start, size_t stop) {
    size_t begin = points[start].location;
    const codepoint *last = &points[stop - 1];
    return copy_range(text, begin, last->location + last->length);
}

static size_t contraction(const codepoint *points, size_t count, size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

static int add_point_slice(string_list *pieces, const char *text,
                           const codepoint *points, size_t start, size_t stop) {
    char *piece = slice_points(text, points, start, stop);
    if (!piece || !strings_push(pieces, piece)) {
        free(piece);
        return 0;
    }
    return 1;
}

static int pretokenize(const char *utf8, string_list *pieces,
                       char *error, size_t error_size) {
    char *text = normalize_nfc(utf8, error, error_size);
    if (!text) return 0;
    size_t count = 0;
    codepoint *points = codepoints(text, &count, error, error_size);
    if (!points) {
        free(text);
        return 0;
    }
    size_t index = 0;
    while (index < count) {
        size_t contracted = contraction(points, count, index);
        if (contracted) {
            if (!add_point_slice(pieces, text, points, index,
                                 index + contracted)) goto memory_failure;
            index += contracted;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (is_letter(value)) {
            /* Already at the first letter. */
        } else if (value != '\r' && value != '\n' && !is_number(value) &&
                   index + 1 < count && is_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < count && is_letter(points[stop].value)) stop++;
            if (!add_point_slice(pieces, text, points, index, stop))
                goto memory_failure;
            index = stop;
            continue;
        }
        if (is_number(value)) {
            if (!add_point_slice(pieces, text, points, index, index + 1))
                goto memory_failure;
            index++;
            continue;
        }
        size_t punctuation_start = index +
            (value == ' ' && index + 1 < count &&
             !is_space(points[index + 1].value) &&
             !is_letter(points[index + 1].value) &&
             !is_number(points[index + 1].value));
        size_t stop = punctuation_start;
        while (stop < count && !is_space(points[stop].value) &&
               !is_letter(points[stop].value) &&
               !is_number(points[stop].value)) stop++;
        if (stop > punctuation_start) {
            while (stop < count &&
                   (points[stop].value == '\r' || points[stop].value == '\n'))
                stop++;
            if (!add_point_slice(pieces, text, points, index, stop))
                goto memory_failure;
            index = stop;
            continue;
        }
        if (is_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < count &&
                   is_space(points[whitespace_end].value)) whitespace_end++;
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++)
                if (points[cursor].value == '\r' ||
                    points[cursor].value == '\n')
                    newline_end = (ptrdiff_t)cursor + 1;
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == count) piece_end = whitespace_end;
            else if (whitespace_end - index > 1) piece_end = whitespace_end - 1;
            else piece_end = index + 1;
            if (!add_point_slice(pieces, text, points, index, piece_end))
                goto memory_failure;
            index = piece_end;
            continue;
        }
        tok_fail(error, error_size, "unable to pre-tokenize input");
        free(points);
        free(text);
        return 0;
    }
    free(points);
    free(text);
    return 1;

memory_failure:
    tok_fail(error, error_size, "out of memory pre-tokenizing prompt");
    free(points);
    free(text);
    return 0;
}

static char *pair_key(const char *left, const char *right) {
    size_t left_length = strlen(left), right_length = strlen(right);
    if (left_length > SIZE_MAX - right_length - 2) return NULL;
    char *key = malloc(left_length + right_length + 2);
    if (!key) return NULL;
    memcpy(key, left, left_length);
    key[left_length] = '\x1f';
    memcpy(key + left_length + 1, right, right_length + 1);
    return key;
}

static char *join_symbols(const char *left, const char *right) {
    size_t left_length = strlen(left), right_length = strlen(right);
    if (left_length > SIZE_MAX - right_length - 1) return NULL;
    char *result = malloc(left_length + right_length + 1);
    if (!result) return NULL;
    memcpy(result, left, left_length);
    memcpy(result + left_length, right, right_length + 1);
    return result;
}

static int split_codepoint_symbols(const char *text, string_list *symbols) {
    size_t length = strlen(text);
    if (length > INT32_MAX) return 0;
    int32_t offset = 0;
    while (offset < (int32_t)length) {
        int32_t start = offset;
        UChar32 value;
        U8_NEXT(text, offset, (int32_t)length, value);
        if (value < 0) return 0;
        char *symbol = copy_range(text, (size_t)start, (size_t)offset);
        if (!symbol || !strings_push(symbols, symbol)) {
            free(symbol);
            return 0;
        }
    }
    return 1;
}

static int bpe(const h3_tokenizer *tokenizer, const char *piece,
               id_list *output, char *error, size_t error_size) {
    size_t bytes = strlen(piece), encoded_length = 0;
    for (size_t index = 0; index < bytes; index++) {
        size_t item = strlen(tokenizer->byte_encoder[(unsigned char)piece[index]]);
        if (encoded_length > SIZE_MAX - item - 1) goto memory_failure;
        encoded_length += item;
    }
    char *encoded = malloc(encoded_length + 1);
    if (!encoded) goto memory_failure;
    size_t used = 0;
    for (size_t index = 0; index < bytes; index++) {
        const char *item = tokenizer->byte_encoder[(unsigned char)piece[index]];
        size_t length = strlen(item);
        memcpy(encoded + used, item, length);
        used += length;
    }
    encoded[used] = '\0';
    string_list symbols = {0};
    if (!split_codepoint_symbols(encoded, &symbols)) {
        free(encoded);
        goto memory_failure;
    }
    free(encoded);
    while (symbols.count > 1) {
        int found = 0;
        uint32_t best_rank = UINT32_MAX;
        size_t best = 0;
        for (size_t index = 0; index + 1 < symbols.count; index++) {
            char *key = pair_key(symbols.items[index], symbols.items[index + 1]);
            if (!key) {
                strings_free(&symbols);
                goto memory_failure;
            }
            const token_map_slot *rank = map_find(&tokenizer->merge_ranks, key);
            free(key);
            if (rank && (!found || rank->value < best_rank)) {
                found = 1;
                best_rank = rank->value;
                best = index;
            }
        }
        if (!found) break;
        const char *left = symbols.items[best];
        const char *right = symbols.items[best + 1];
        string_list merged = {0};
        for (size_t index = 0; index < symbols.count;) {
            char *item;
            if (index + 1 < symbols.count &&
                !strcmp(symbols.items[index], left) &&
                !strcmp(symbols.items[index + 1], right)) {
                item = join_symbols(symbols.items[index], symbols.items[index + 1]);
                index += 2;
            } else {
                item = strdup(symbols.items[index++]);
            }
            if (!item || !strings_push(&merged, item)) {
                free(item);
                strings_free(&merged);
                strings_free(&symbols);
                goto memory_failure;
            }
        }
        strings_free(&symbols);
        symbols = merged;
    }
    for (size_t index = 0; index < symbols.count; index++) {
        const token_map_slot *token = map_find(&tokenizer->vocab,
                                               symbols.items[index]);
        if (!token) {
            tok_fail(error, error_size,
                     "BPE symbol is absent from vocabulary: %s",
                     symbols.items[index]);
            strings_free(&symbols);
            return 0;
        }
        if (!ids_push(output, token->value)) {
            strings_free(&symbols);
            goto memory_failure;
        }
    }
    strings_free(&symbols);
    return 1;

memory_failure:
    tok_fail(error, error_size, "out of memory applying BPE");
    return 0;
}

static int encode_plain(const h3_tokenizer *tokenizer, const char *text,
                        id_list *output, char *error, size_t error_size) {
    string_list pieces = {0};
    if (!pretokenize(text, &pieces, error, error_size)) {
        strings_free(&pieces);
        return 0;
    }
    for (size_t index = 0; index < pieces.count; index++) {
        if (!bpe(tokenizer, pieces.items[index], output, error, error_size)) {
            strings_free(&pieces);
            return 0;
        }
    }
    strings_free(&pieces);
    return 1;
}

static int json_bool(const h3_json_value *object, const char *key,
                     int *value) {
    return h3_json_boolean_value(h3_json_get(object, key), value);
}

static int alternative_compare(const void *left, const void *right) {
    const char *a = *(const char *const *)left;
    const char *b = *(const char *const *)right;
    size_t a_length = strlen(a), b_length = strlen(b);
    if (a_length > b_length) return -1;
    if (a_length < b_length) return 1;
    return strcmp(a, b);
}

static char *codepoint_utf8(uint32_t value) {
    char buffer[U8_MAX_LENGTH + 1];
    int32_t offset = 0;
    U8_APPEND_UNSAFE(buffer, offset, (UChar32)value);
    buffer[offset] = '\0';
    return strdup(buffer);
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        tok_fail(error, error_size, "tokenizer path is required");
        return NULL;
    }
    h3_json_value *document = h3_json_parse_file(path, error, error_size);
    if (!document) return NULL;
    const h3_json_value *model = h3_json_get(document, "model");
    const h3_json_value *normalizer = h3_json_get(document, "normalizer");
    const h3_json_value *vocab = h3_json_get(model, "vocab");
    const h3_json_value *merges = h3_json_get(model, "merges");
    const char *model_type = h3_json_string_value(h3_json_get(model, "type"));
    const char *normalizer_type = h3_json_string_value(
        h3_json_get(normalizer, "type"));
    if (!model_type || strcmp(model_type, "BPE") ||
        h3_json_get_type(h3_json_get(model, "unk_token")) != H3_JSON_NULL ||
        !normalizer_type || strcmp(normalizer_type, "NFC") ||
        h3_json_get_type(vocab) != H3_JSON_OBJECT ||
        h3_json_get_type(merges) != H3_JSON_ARRAY) {
        tok_fail(error, error_size, "unexpected tokenizer specification");
        h3_json_free(document);
        return NULL;
    }
    const h3_json_value *added = h3_json_get(document, "added_tokens");
    if (added && h3_json_get_type(added) != H3_JSON_ARRAY) {
        tok_fail(error, error_size, "invalid added-token table");
        h3_json_free(document);
        return NULL;
    }
    size_t vocab_count = h3_json_size(vocab);
    size_t merge_count = h3_json_size(merges);
    size_t added_count = added ? h3_json_size(added) : 0;
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer || !map_init(&tokenizer->vocab, vocab_count, 0) ||
        !map_init(&tokenizer->merge_ranks, merge_count, 1) ||
        !map_init(&tokenizer->added_tokens, added_count, 0)) {
        tok_fail(error, error_size, "out of memory loading tokenizer");
        if (tokenizer) h3_tokenizer_free(tokenizer);
        h3_json_free(document);
        return NULL;
    }
    tokenizer->document = document;
    uint32_t maximum_id = 0;
    for (size_t index = 0; index < vocab_count; index++) {
        const char *symbol = h3_json_object_key(vocab, index);
        int64_t identifier;
        if (!symbol || !h3_json_i64_value(
                h3_json_object_value(vocab, index), &identifier) ||
            identifier < 0 || identifier > UINT32_MAX ||
            !map_insert(&tokenizer->vocab, symbol, (uint32_t)identifier)) {
            tok_fail(error, error_size, "invalid tokenizer vocabulary");
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
        if ((uint32_t)identifier > maximum_id) maximum_id = (uint32_t)identifier;
    }
    for (size_t index = 0; index < added_count; index++) {
        const h3_json_value *item = h3_json_at(added, index);
        int64_t identifier;
        if (!h3_json_i64_value(h3_json_get(item, "id"), &identifier) ||
            identifier < 0 || identifier > UINT32_MAX) {
            tok_fail(error, error_size, "invalid added-token identifier");
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
        if ((uint32_t)identifier > maximum_id) maximum_id = (uint32_t)identifier;
    }
    tokenizer->inverse_count = (size_t)maximum_id + 1;
    tokenizer->inverse_vocab = calloc(
        tokenizer->inverse_count, sizeof(*tokenizer->inverse_vocab));
    tokenizer->inverse_added = calloc(
        tokenizer->inverse_count, sizeof(*tokenizer->inverse_added));
    tokenizer->added_alternatives = calloc(
        added_count ? added_count : 1, sizeof(*tokenizer->added_alternatives));
    if (!tokenizer->inverse_vocab || !tokenizer->inverse_added ||
        !tokenizer->added_alternatives) {
        tok_fail(error, error_size, "out of memory indexing tokenizer");
        h3_tokenizer_free(tokenizer);
        return NULL;
    }
    for (size_t index = 0; index < vocab_count; index++) {
        int64_t identifier;
        (void)h3_json_i64_value(h3_json_object_value(vocab, index), &identifier);
        tokenizer->inverse_vocab[(size_t)identifier] =
            h3_json_object_key(vocab, index);
    }
    for (size_t index = 0; index < merge_count; index++) {
        const h3_json_value *item = h3_json_at(merges, index);
        const char *entry = h3_json_string_value(item);
        char *key = NULL;
        if (entry) {
            const char *separator = strchr(entry, ' ');
            if (separator) {
                char *left = copy_range(entry, 0, (size_t)(separator - entry));
                key = left ? pair_key(left, separator + 1) : NULL;
                free(left);
            }
        } else if (h3_json_get_type(item) == H3_JSON_ARRAY &&
                   h3_json_size(item) == 2) {
            const char *left = h3_json_string_value(h3_json_at(item, 0));
            const char *right = h3_json_string_value(h3_json_at(item, 1));
            if (left && right) key = pair_key(left, right);
        }
        if (!key || index > UINT32_MAX ||
            !map_insert(&tokenizer->merge_ranks, key, (uint32_t)index)) {
            free(key);
            tok_fail(error, error_size, "invalid tokenizer merge");
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
    }
    for (size_t index = 0; index < added_count; index++) {
        const h3_json_value *item = h3_json_at(added, index);
        int single_word, lstrip, rstrip, normalized;
        int64_t identifier;
        const char *content = h3_json_string_value(h3_json_get(item, "content"));
        if (!content || !h3_json_i64_value(h3_json_get(item, "id"), &identifier) ||
            !json_bool(item, "single_word", &single_word) ||
            !json_bool(item, "lstrip", &lstrip) ||
            !json_bool(item, "rstrip", &rstrip) ||
            !json_bool(item, "normalized", &normalized) || single_word ||
            lstrip || rstrip || normalized ||
            !map_insert(&tokenizer->added_tokens, content,
                        (uint32_t)identifier)) {
            tok_fail(error, error_size, "unsupported added-token policy");
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
        tokenizer->inverse_added[(size_t)identifier] = content;
        tokenizer->added_alternatives[index] = content;
    }
    tokenizer->added_count = added_count;
    qsort(tokenizer->added_alternatives, added_count,
          sizeof(*tokenizer->added_alternatives), alternative_compare);

    for (size_t index = 0; index < 324; index++) tokenizer->byte_decoder[index] = -1;
    unsigned extra = 0;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t point = visible ? byte : 256 + extra++;
        tokenizer->byte_encoder[byte] = codepoint_utf8(point);
        if (!tokenizer->byte_encoder[byte]) {
            tok_fail(error, error_size, "out of memory building byte encoder");
            h3_tokenizer_free(tokenizer);
            return NULL;
        }
        tokenizer->byte_decoder[point] = (int16_t)byte;
    }
    return tokenizer;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    for (size_t index = 0; index < 256; index++) free(tokenizer->byte_encoder[index]);
    free(tokenizer->added_alternatives);
    free(tokenizer->inverse_added);
    free(tokenizer->inverse_vocab);
    map_free(&tokenizer->added_tokens);
    map_free(&tokenizer->merge_ranks);
    map_free(&tokenizer->vocab);
    h3_json_free(tokenizer->document);
    free(tokenizer);
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    id_list output = {0};
    size_t text_length = strlen(utf8), start = 0;
    while (start < text_length) {
        const char *best = NULL;
        size_t best_location = SIZE_MAX, best_length = 0;
        for (size_t index = 0; index < tokenizer->added_count; index++) {
            const char *candidate = tokenizer->added_alternatives[index];
            const char *found = strstr(utf8 + start, candidate);
            if (!found) continue;
            size_t location = (size_t)(found - utf8);
            size_t length = strlen(candidate);
            if (!best || location < best_location ||
                (location == best_location && length > best_length)) {
                best = candidate;
                best_location = location;
                best_length = length;
            }
        }
        if (!best) break;
        if (best_location > start) {
            char *plain = copy_range(utf8, start, best_location);
            if (!plain || !encode_plain(tokenizer, plain, &output,
                                        error, error_size)) {
                free(plain);
                free(output.items);
                return 0;
            }
            free(plain);
        }
        const token_map_slot *added = map_find(&tokenizer->added_tokens, best);
        if (!added || !ids_push(&output, added->value)) {
            free(output.items);
            tok_fail(error, error_size, "out of memory encoding prompt");
            return 0;
        }
        start = best_location + best_length;
    }
    if (start < text_length) {
        if (!encode_plain(tokenizer, utf8 + start, &output,
                          error, error_size)) {
            free(output.items);
            return 0;
        }
    }
    if (!output.count && pad_empty && !ids_push(&output, H3_PAD_TOKEN_ID)) {
        free(output.items);
        tok_fail(error, error_size, "out of memory encoding empty prompt");
        return 0;
    }
    *ids = output.items;
    *count = output.count;
    return 1;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

static int append_bytes(char **buffer, size_t *length, size_t *capacity,
                        const char *bytes, size_t count) {
    if (*length > SIZE_MAX - count - 1) return 0;
    size_t needed = *length + count + 1;
    if (needed > *capacity) {
        size_t next = *capacity ? *capacity : 64;
        while (next < needed) {
            if (next > SIZE_MAX / 2) return 0;
            next *= 2;
        }
        char *grown = realloc(*buffer, next);
        if (!grown) return 0;
        *buffer = grown;
        *capacity = next;
    }
    memcpy(*buffer + *length, bytes, count);
    *length += count;
    (*buffer)[*length] = '\0';
    return 1;
}

static int flush_decoded(char **result, size_t *result_length,
                         size_t *result_capacity, unsigned char *bytes,
                         size_t *byte_count) {
    if (!*byte_count) return 1;
    UErrorCode status = U_ZERO_ERROR;
    int32_t units = 0;
    u_strFromUTF8(NULL, 0, &units, (const char *)bytes, (int32_t)*byte_count,
                  &status);
    if (status == U_BUFFER_OVERFLOW_ERROR) {
        if (!append_bytes(result, result_length, result_capacity,
                          (const char *)bytes, *byte_count)) return 0;
    } else {
        static const char replacement[] = "\xef\xbf\xbd";
        if (!append_bytes(result, result_length, result_capacity,
                          replacement, sizeof(replacement) - 1)) return 0;
    }
    *byte_count = 0;
    return 1;
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer || (!ids && count)) return NULL;
    char *result = NULL;
    size_t result_length = 0, result_capacity = 0;
    unsigned char *bytes = NULL;
    size_t byte_count = 0, byte_capacity = 0;
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if (identifier >= tokenizer->inverse_count) {
            tok_fail(error, error_size, "token ID is out of range");
            goto failed;
        }
        const char *added = tokenizer->inverse_added[identifier];
        if (added) {
            if (!flush_decoded(&result, &result_length, &result_capacity,
                               bytes, &byte_count) ||
                !append_bytes(&result, &result_length, &result_capacity,
                              added, strlen(added))) goto memory_failure;
            continue;
        }
        const char *symbol = tokenizer->inverse_vocab[identifier];
        if (!symbol) {
            tok_fail(error, error_size, "unknown token ID");
            goto failed;
        }
        size_t length = strlen(symbol);
        if (length > INT32_MAX) goto memory_failure;
        int32_t offset = 0;
        while (offset < (int32_t)length) {
            UChar32 point;
            U8_NEXT(symbol, offset, (int32_t)length, point);
            if (point < 0 || point >= 324 || tokenizer->byte_decoder[point] < 0) {
                tok_fail(error, error_size, "invalid byte-level token");
                goto failed;
            }
            if (byte_count == byte_capacity) {
                size_t next = byte_capacity ? byte_capacity * 2 : 64;
                if (next < byte_capacity) goto memory_failure;
                unsigned char *grown = realloc(bytes, next);
                if (!grown) goto memory_failure;
                bytes = grown;
                byte_capacity = next;
            }
            bytes[byte_count++] = (unsigned char)tokenizer->byte_decoder[point];
        }
    }
    if (!flush_decoded(&result, &result_length, &result_capacity,
                       bytes, &byte_count)) goto memory_failure;
    free(bytes);
    if (!result) result = strdup("");
    if (!result) goto memory_failure_no_bytes;
    return result;

memory_failure:
    tok_fail(error, error_size, "out of memory decoding tokens");
failed:
    free(bytes);
memory_failure_no_bytes:
    free(result);
    return NULL;
}
