#include <stdio.h>

struct unpacked {
        char c;
        long l;
};

struct packed {
        char c;
        long l;
} __attribute__ ((packed));

int main(void) {
        printf("unpacked = %ld\n", sizeof(struct unpacked));
        printf("packed   = %ld\n", sizeof(struct packed));
        return 0;
}

