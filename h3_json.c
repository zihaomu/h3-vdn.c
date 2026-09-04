#include "h3_json.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define H3_JSON_MAX_FILE (32u * 1024u * 1024u)
#define H3_JSON_MAX_DEPTH 128

typedef struct {
    char *key;
    h3_json_value *value;
} h3_json_member;

struct h3_json_value {
    h3_json_type type;
    union {
        int boolean;
        double number;
        char *string;
        struct {
            h3_json_value **items;
            size_t count;
        } array;
        struct {
            h3_json_member *members;
            size_t count;
        } object;
    } as;
};

typedef struct {
    const char *text;
    size_t length;
    size_t offset;
    unsigned depth;
    char *error;
    size_t error_size;
} h3_json_parser;

static int h3_json_fail(h3_json_parser *parser, const char *message) {
    if (parser->error && parser->error_size) {
        size_t line = 1;
        size_t column = 1;
        for (size_t index = 0; index < parser->offset &&
                               index < parser->length; index++) {
            if (parser->text[index] == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        snprintf(parser->error, parser->error_size,
                 "JSON line %zu column %zu: %s", line, column, message);
    }
    return 0;
}

static void h3_json_ws(h3_json_parser *parser) {
    while (parser->offset < parser->length) {
        char value = parser->text[parser->offset];
        if (value != ' ' && value != '\t' && value != '\r' && value != '\n')
            break;
        parser->offset++;
    }
}

static h3_json_value *h3_json_new(h3_json_type type,
                                  h3_json_parser *parser) {
    h3_json_value *value = calloc(1, sizeof(*value));
    if (!value) {
        h3_json_fail(parser, "out of memory");
        return NULL;
    }
    value->type = type;
    return value;
}

static int h3_json_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static int h3_json_unicode_escape(h3_json_parser *parser,
                                  uint32_t *codepoint) {
    if (parser->length - parser->offset < 4)
        return h3_json_fail(parser, "short Unicode escape");
    uint32_t value = 0;
    for (int index = 0; index < 4; index++) {
        int digit = h3_json_hex(parser->text[parser->offset++]);
        if (digit < 0) return h3_json_fail(parser, "invalid Unicode escape");
        value = value * 16 + (uint32_t)digit;
    }
    *codepoint = value;
    return 1;
}

static int h3_json_append_utf8(char *output, size_t *length,
                               uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output[(*length)++] = (char)codepoint;
    } else if (codepoint <= 0x7ff) {
        output[(*length)++] = (char)(0xc0u | (codepoint >> 6));
        output[(*length)++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0xffff) {
        output[(*length)++] = (char)(0xe0u | (codepoint >> 12));
        output[(*length)++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        output[(*length)++] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (codepoint <= 0x10ffff) {
        output[(*length)++] = (char)(0xf0u | (codepoint >> 18));
        output[(*length)++] = (char)(0x80u | ((codepoint >> 12) & 0x3fu));
        output[(*length)++] = (char)(0x80u | ((codepoint >> 6) & 0x3fu));
        output[(*length)++] = (char)(0x80u | (codepoint & 0x3fu));
    } else {
        return 0;
    }
    return 1;
}

static char *h3_json_parse_string(h3_json_parser *parser) {
    h3_json_ws(parser);
    if (parser->offset >= parser->length ||
        parser->text[parser->offset] != '"') {
        h3_json_fail(parser, "expected string");
        return NULL;
    }
    parser->offset++;
    size_t capacity = parser->length - parser->offset + 1;
    char *output = malloc(capacity);
    if (!output) {
        h3_json_fail(parser, "out of memory");
        return NULL;
    }
    size_t used = 0;
    while (parser->offset < parser->length) {
        unsigned char value = (unsigned char)parser->text[parser->offset++];
        if (value == '"') {
            output[used] = '\0';
            return output;
        }
        if (value < 0x20) goto malformed;
        if (value != '\\') {
            output[used++] = (char)value;
            continue;
        }
        if (parser->offset >= parser->length) goto malformed;
        value = (unsigned char)parser->text[parser->offset++];
        switch (value) {
            case '"': output[used++] = '"'; break;
            case '\\': output[used++] = '\\'; break;
            case '/': output[used++] = '/'; break;
            case 'b': output[used++] = '\b'; break;
            case 'f': output[used++] = '\f'; break;
            case 'n': output[used++] = '\n'; break;
            case 'r': output[used++] = '\r'; break;
            case 't': output[used++] = '\t'; break;
            case 'u': {
                uint32_t codepoint;
                if (!h3_json_unicode_escape(parser, &codepoint)) {
                    free(output);
                    return NULL;
                }
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if (parser->length - parser->offset < 6 ||
                        parser->text[parser->offset] != '\\' ||
                        parser->text[parser->offset + 1] != 'u')
                        goto malformed;
                    parser->offset += 2;
                    uint32_t low;
                    if (!h3_json_unicode_escape(parser, &low)) {
                        free(output);
                        return NULL;
                    }
                    if (low < 0xdc00 || low > 0xdfff) goto malformed;
                    codepoint = UINT32_C(0x10000) +
                        ((codepoint - 0xd800u) << 10) + (low - 0xdc00u);
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                    goto malformed;
                }
                if (!h3_json_append_utf8(output, &used, codepoint))
                    goto malformed;
                break;
            }
            default: goto malformed;
        }
    }

malformed:
    free(output);
    h3_json_fail(parser, "malformed string escape or unterminated string");
    return NULL;
}

static h3_json_value *h3_json_parse_value(h3_json_parser *parser);

static int h3_json_enter(h3_json_parser *parser) {
    if (parser->depth >= H3_JSON_MAX_DEPTH)
        return h3_json_fail(parser, "maximum nesting depth exceeded");
    parser->depth++;
    return 1;
}

static h3_json_value *h3_json_parse_array(h3_json_parser *parser) {
    if (!h3_json_enter(parser)) return NULL;
    parser->offset++;
    h3_json_value *array = h3_json_new(H3_JSON_ARRAY, parser);
    if (!array) goto failed;
    h3_json_ws(parser);
    if (parser->offset < parser->length &&
        parser->text[parser->offset] == ']') {
        parser->offset++;
        parser->depth--;
        return array;
    }
    for (;;) {
        h3_json_value *item = h3_json_parse_value(parser);
        if (!item) goto failed_value;
        size_t count = array->as.array.count;
        if (count == SIZE_MAX / sizeof(*array->as.array.items)) {
            h3_json_free(item);
            h3_json_fail(parser, "array is too large");
            goto failed_value;
        }
        h3_json_value **items = realloc(
            array->as.array.items, (count + 1) * sizeof(*items));
        if (!items) {
            h3_json_free(item);
            h3_json_fail(parser, "out of memory");
            goto failed_value;
        }
        array->as.array.items = items;
        array->as.array.items[count] = item;
        array->as.array.count++;
        h3_json_ws(parser);
        if (parser->offset >= parser->length) {
            h3_json_fail(parser, "unterminated array");
            goto failed_value;
        }
        char delimiter = parser->text[parser->offset++];
        if (delimiter == ']') break;
        if (delimiter != ',') {
            h3_json_fail(parser, "expected array comma");
            goto failed_value;
        }
    }
    parser->depth--;
    return array;

failed_value:
    h3_json_free(array);
failed:
    parser->depth--;
    return NULL;
}

static int h3_json_duplicate_key(const h3_json_value *object,
                                 const char *key) {
    for (size_t index = 0; index < object->as.object.count; index++)
        if (!strcmp(object->as.object.members[index].key, key)) return 1;
    return 0;
}

static h3_json_value *h3_json_parse_object(h3_json_parser *parser) {
    if (!h3_json_enter(parser)) return NULL;
    parser->offset++;
    h3_json_value *object = h3_json_new(H3_JSON_OBJECT, parser);
    if (!object) goto failed;
    h3_json_ws(parser);
    if (parser->offset < parser->length &&
        parser->text[parser->offset] == '}') {
        parser->offset++;
        parser->depth--;
        return object;
    }
    for (;;) {
        char *key = h3_json_parse_string(parser);
        if (!key) goto failed_value;
        if (h3_json_duplicate_key(object, key)) {
            free(key);
            h3_json_fail(parser, "duplicate object key");
            goto failed_value;
        }
        h3_json_ws(parser);
        if (parser->offset >= parser->length ||
            parser->text[parser->offset++] != ':') {
            free(key);
            h3_json_fail(parser, "expected object colon");
            goto failed_value;
        }
        h3_json_value *value = h3_json_parse_value(parser);
        if (!value) {
            free(key);
            goto failed_value;
        }
        size_t count = object->as.object.count;
        if (count == SIZE_MAX / sizeof(*object->as.object.members)) {
            free(key);
            h3_json_free(value);
            h3_json_fail(parser, "object is too large");
            goto failed_value;
        }
        h3_json_member *members = realloc(
            object->as.object.members, (count + 1) * sizeof(*members));
        if (!members) {
            free(key);
            h3_json_free(value);
            h3_json_fail(parser, "out of memory");
            goto failed_value;
        }
        object->as.object.members = members;
        object->as.object.members[count] = (h3_json_member){key, value};
        object->as.object.count++;
        h3_json_ws(parser);
        if (parser->offset >= parser->length) {
            h3_json_fail(parser, "unterminated object");
            goto failed_value;
        }
        char delimiter = parser->text[parser->offset++];
        if (delimiter == '}') break;
        if (delimiter != ',') {
            h3_json_fail(parser, "expected object comma");
            goto failed_value;
        }
    }
    parser->depth--;
    return object;

failed_value:
    h3_json_free(object);
failed:
    parser->depth--;
    return NULL;
}

static int h3_json_literal(h3_json_parser *parser, const char *literal) {
    size_t length = strlen(literal);
    if (parser->length - parser->offset < length ||
        memcmp(parser->text + parser->offset, literal, length))
        return h3_json_fail(parser, "invalid JSON literal");
    parser->offset += length;
    return 1;
}

static h3_json_value *h3_json_parse_number(h3_json_parser *parser) {
    size_t start = parser->offset;
    if (parser->text[parser->offset] == '-') parser->offset++;
    if (parser->offset >= parser->length)
        goto malformed;
    if (parser->text[parser->offset] == '0') {
        parser->offset++;
        if (parser->offset < parser->length &&
            parser->text[parser->offset] >= '0' &&
            parser->text[parser->offset] <= '9') goto malformed;
    } else if (parser->text[parser->offset] >= '1' &&
               parser->text[parser->offset] <= '9') {
        while (parser->offset < parser->length &&
               parser->text[parser->offset] >= '0' &&
               parser->text[parser->offset] <= '9') parser->offset++;
    } else {
        goto malformed;
    }
    if (parser->offset < parser->length &&
        parser->text[parser->offset] == '.') {
        parser->offset++;
        size_t digits = parser->offset;
        while (parser->offset < parser->length &&
               parser->text[parser->offset] >= '0' &&
               parser->text[parser->offset] <= '9') parser->offset++;
        if (digits == parser->offset) goto malformed;
    }
    if (parser->offset < parser->length &&
        (parser->text[parser->offset] == 'e' ||
         parser->text[parser->offset] == 'E')) {
        parser->offset++;
        if (parser->offset < parser->length &&
            (parser->text[parser->offset] == '+' ||
             parser->text[parser->offset] == '-')) parser->offset++;
        size_t digits = parser->offset;
        while (parser->offset < parser->length &&
               parser->text[parser->offset] >= '0' &&
               parser->text[parser->offset] <= '9') parser->offset++;
        if (digits == parser->offset) goto malformed;
    }

    size_t length = parser->offset - start;
    char *copy = malloc(length + 1);
    if (!copy) {
        h3_json_fail(parser, "out of memory");
        return NULL;
    }
    memcpy(copy, parser->text + start, length);
    copy[length] = '\0';
    char *end = NULL;
    errno = 0;
    double number = strtod(copy, &end);
    int valid = !errno && end == copy + length && isfinite(number);
    free(copy);
    if (!valid) {
        h3_json_fail(parser, "number is outside the supported range");
        return NULL;
    }
    h3_json_value *value = h3_json_new(H3_JSON_NUMBER, parser);
    if (value) value->as.number = number;
    return value;

malformed:
    parser->offset = start;
    h3_json_fail(parser, "malformed number");
    return NULL;
}

static h3_json_value *h3_json_parse_value(h3_json_parser *parser) {
    h3_json_ws(parser);
    if (parser->offset >= parser->length) {
        h3_json_fail(parser, "missing value");
        return NULL;
    }
    char first = parser->text[parser->offset];
    if (first == '{') return h3_json_parse_object(parser);
    if (first == '[') return h3_json_parse_array(parser);
    if (first == '"') {
        char *string = h3_json_parse_string(parser);
        if (!string) return NULL;
        h3_json_value *value = h3_json_new(H3_JSON_STRING, parser);
        if (!value) {
            free(string);
            return NULL;
        }
        value->as.string = string;
        return value;
    }
    if (first == '-' || (first >= '0' && first <= '9'))
        return h3_json_parse_number(parser);

    h3_json_type type;
    int boolean = 0;
    if (first == 't') {
        if (!h3_json_literal(parser, "true")) return NULL;
        type = H3_JSON_BOOLEAN;
        boolean = 1;
    } else if (first == 'f') {
        if (!h3_json_literal(parser, "false")) return NULL;
        type = H3_JSON_BOOLEAN;
    } else if (first == 'n') {
        if (!h3_json_literal(parser, "null")) return NULL;
        type = H3_JSON_NULL;
    } else {
        h3_json_fail(parser, "unexpected token");
        return NULL;
    }
    h3_json_value *value = h3_json_new(type, parser);
    if (value) value->as.boolean = boolean;
    return value;
}

h3_json_value *h3_json_parse(const char *text, size_t length,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!text) {
        if (error && error_size) snprintf(error, error_size, "JSON input is null");
        return NULL;
    }
    h3_json_parser parser = {text, length, 0, 0, error, error_size};
    h3_json_value *value = h3_json_parse_value(&parser);
    if (!value) return NULL;
    h3_json_ws(&parser);
    if (parser.offset != parser.length) {
        h3_json_fail(&parser, "trailing data");
        h3_json_free(value);
        return NULL;
    }
    return value;
}

h3_json_value *h3_json_parse_file(const char *path,
                                  char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path) {
        if (error && error_size) snprintf(error, error_size, "JSON path is required");
        return NULL;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return NULL;
    }
    struct stat status;
    if (fstat(descriptor, &status) != 0 || status.st_size < 0 ||
        (uint64_t)status.st_size > H3_JSON_MAX_FILE) {
        if (error && error_size)
            snprintf(error, error_size, "%s: invalid or oversized JSON file", path);
        close(descriptor);
        return NULL;
    }
    size_t length = (size_t)status.st_size;
    char *text = malloc(length + 1);
    if (!text) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        close(descriptor);
        return NULL;
    }
    size_t complete = 0;
    while (complete < length) {
        ssize_t count = read(descriptor, text + complete, length - complete);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            if (error && error_size)
                snprintf(error, error_size, "%s: cannot read JSON: %s", path,
                         count < 0 ? strerror(errno) : "unexpected end of file");
            free(text);
            close(descriptor);
            return NULL;
        }
        complete += (size_t)count;
    }
    close(descriptor);
    text[length] = '\0';
    h3_json_value *value = h3_json_parse(text, length, error, error_size);
    if (!value && error && error_size) {
        size_t used = strlen(error);
        if (used + strlen(path) + 3 < error_size) {
            memmove(error + strlen(path) + 2, error, used + 1);
            memcpy(error, path, strlen(path));
            memcpy(error + strlen(path), ": ", 2);
        }
    }
    free(text);
    return value;
}

void h3_json_free(h3_json_value *value) {
    if (!value) return;
    if (value->type == H3_JSON_STRING) {
        free(value->as.string);
    } else if (value->type == H3_JSON_ARRAY) {
        for (size_t index = 0; index < value->as.array.count; index++)
            h3_json_free(value->as.array.items[index]);
        free(value->as.array.items);
    } else if (value->type == H3_JSON_OBJECT) {
        for (size_t index = 0; index < value->as.object.count; index++) {
            free(value->as.object.members[index].key);
            h3_json_free(value->as.object.members[index].value);
        }
        free(value->as.object.members);
    }
    free(value);
}

h3_json_type h3_json_get_type(const h3_json_value *value) {
    return value ? value->type : H3_JSON_NULL;
}

size_t h3_json_size(const h3_json_value *value) {
    if (!value) return 0;
    if (value->type == H3_JSON_ARRAY) return value->as.array.count;
    if (value->type == H3_JSON_OBJECT) return value->as.object.count;
    return 0;
}

const h3_json_value *h3_json_at(const h3_json_value *array, size_t index) {
    if (!array || array->type != H3_JSON_ARRAY ||
        index >= array->as.array.count) return NULL;
    return array->as.array.items[index];
}

const char *h3_json_object_key(const h3_json_value *object, size_t index) {
    if (!object || object->type != H3_JSON_OBJECT ||
        index >= object->as.object.count) return NULL;
    return object->as.object.members[index].key;
}

const h3_json_value *h3_json_object_value(const h3_json_value *object,
                                          size_t index) {
    if (!object || object->type != H3_JSON_OBJECT ||
        index >= object->as.object.count) return NULL;
    return object->as.object.members[index].value;
}

const h3_json_value *h3_json_get(const h3_json_value *object,
                                 const char *key) {
    if (!object || object->type != H3_JSON_OBJECT || !key) return NULL;
    for (size_t index = 0; index < object->as.object.count; index++)
        if (!strcmp(object->as.object.members[index].key, key))
            return object->as.object.members[index].value;
    return NULL;
}

const char *h3_json_string_value(const h3_json_value *value) {
    return value && value->type == H3_JSON_STRING ? value->as.string : NULL;
}

int h3_json_boolean_value(const h3_json_value *value, int *result) {
    if (!value || value->type != H3_JSON_BOOLEAN || !result) return 0;
    *result = value->as.boolean;
    return 1;
}

int h3_json_i64_value(const h3_json_value *value, int64_t *result) {
    if (!value || value->type != H3_JSON_NUMBER || !result ||
        value->as.number < -9223372036854775808.0 ||
        value->as.number >= 9223372036854775808.0 ||
        trunc(value->as.number) != value->as.number) return 0;
    *result = (int64_t)value->as.number;
    return 1;
}

int h3_json_f64_value(const h3_json_value *value, double *result) {
    if (!value || value->type != H3_JSON_NUMBER || !result) return 0;
    *result = value->as.number;
    return 1;
}
