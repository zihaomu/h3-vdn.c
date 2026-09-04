#include "h3_json.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    char error[256];
    const char document[] =
        "{\"name\":\"VDN\\u002dH3\",\"ok\":true,\"values\":[2,-1.5,null],"
        "\"nested\":{\"emoji\":\"\\ud83d\\ude80\"}}";
    h3_json_value *root = h3_json_parse(document, strlen(document),
                                         error, sizeof(error));
    CHECK(root != NULL);
    CHECK(h3_json_get_type(root) == H3_JSON_OBJECT);
    CHECK(h3_json_size(root) == 4);
    CHECK(!strcmp(h3_json_object_key(root, 0), "name"));
    CHECK(h3_json_object_value(root, 0) == h3_json_get(root, "name"));
    CHECK(h3_json_object_key(root, 4) == NULL);
    CHECK(!strcmp(h3_json_string_value(h3_json_get(root, "name")), "VDN-H3"));
    int boolean = 0;
    CHECK(h3_json_boolean_value(h3_json_get(root, "ok"), &boolean));
    CHECK(boolean == 1);
    const h3_json_value *values = h3_json_get(root, "values");
    CHECK(h3_json_size(values) == 3);
    int64_t integer = 0;
    CHECK(h3_json_i64_value(h3_json_at(values, 0), &integer));
    CHECK(integer == 2);
    double number = 0.0;
    CHECK(h3_json_f64_value(h3_json_at(values, 1), &number));
    CHECK(number == -1.5);
    const h3_json_value *nested = h3_json_get(root, "nested");
    CHECK(!strcmp(h3_json_string_value(h3_json_get(nested, "emoji")),
                  "\xf0\x9f\x9a\x80"));
    h3_json_free(root);

    const char *invalid[] = {
        "{\"a\":1,\"a\":2}", "[1,]", "01", "\"\\ud800\"", "true false"
    };
    for (size_t index = 0; index < sizeof(invalid) / sizeof(invalid[0]); index++) {
        root = h3_json_parse(invalid[index], strlen(invalid[index]),
                             error, sizeof(error));
        CHECK(root == NULL);
        CHECK(error[0] != '\0');
    }

    puts("JSON tests passed");
    return 0;
}
