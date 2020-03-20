#include "zfp.h"
#include <emscripten.h>
#include <stdlib.h>
#include <string.h>

static float array[120 * 8320 * 3];

EMSCRIPTEN_KEEPALIVE
float* zfpHelper(unsigned char* compressedBuffer)
{
    // ZPF STUFF FROM simple.c EXAMPLE

    zfp_type type;     /* array scalar type */
    zfp_field* field;  /* array meta data */
    zfp_stream* zfp;   /* compressed stream */
    float* buffer;     /* storage for compressed stream */
    size_t bufsize;    /* byte size of compressed buffer */
    bitstream* stream; /* bit stream to write to or read from */

    zfp = zfp_stream_open(NULL);
    type = zfp_type_float;
    // math.ceil(4 * 29.97), 8320, 3 (framecount, vert count, dimension count)
    field = zfp_field_1d(compressedBuffer, type, 120 * 8320 * 3);
    // tolerance 0.001 matching server side python script called main.py
    zfp_stream_set_accuracy(zfp, 0.001);
    
    bufsize = zfp_stream_maximum_size(zfp, field);
    buffer = malloc(bufsize);

    stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    zfp_decompress(zfp, field);

    // CLEANUP

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    memcpy(array, buffer, 120 * 8320 * 3 * 4);

    free(buffer);

    return buffer;
}
