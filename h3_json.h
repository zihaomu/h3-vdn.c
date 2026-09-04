#ifndef H3_JSON_H
#define H3_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    H3_JSON_NULL = 0,
    H3_JSON_BOOLEAN,
    H3_JSON_NUMBER,
    H3_JSON_STRING,
    H3_JSON_ARRAY,
    H3_JSON_OBJECT
} h3_json_type;

typedef struct h3_json_value h3_json_value;

h3_json_value *h3_json_parse(const char *text, size_t length,
                             char *error, size_t error_size);
h3_json_value *h3_json_parse_file(const char *path,
                                  char *error, size_t error_size);
void h3_json_free(h3_json_value *value);

h3_json_type h3_json_get_type(const h3_json_value *value);
size_t h3_json_size(const h3_json_value *value);
const h3_json_value *h3_json_at(const h3_json_value *array, size_t index);
const char *h3_json_object_key(const h3_json_value *object, size_t index);
const h3_json_value *h3_json_object_value(const h3_json_value *object,
                                          size_t index);
const h3_json_value *h3_json_get(const h3_json_value *object,
                                 const char *key);
const char *h3_json_string_value(const h3_json_value *value);
int h3_json_boolean_value(const h3_json_value *value, int *result);
int h3_json_i64_value(const h3_json_value *value, int64_t *result);
int h3_json_f64_value(const h3_json_value *value, double *result);

#endif
