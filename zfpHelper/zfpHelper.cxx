#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <emscripten.h>
#include <zfp.h>

static zfp_field* field;  /* array meta data */
static zfp_stream* zfp;   /* compressed stream */
static bitstream* stream; /* bit stream to write to or read from */

extern "C"
{
    EMSCRIPTEN_KEEPALIVE
    float* zfpHelper(std::string compressedBytes, unsigned int bytesSize)
    {
        // HEAP STUFF

        // js side frees decompressed!
        float* decompressed = (float*)malloc(120 * 8320 * 3 * 8);

        // ZPF STUFF

        // math.ceil(4 * 29.97), 8320, 3 (framecount, vert count, dimension count)
        // C vs Fortran ordering: https://github.com/LLNL/zfp/issues/91
        field = zfp_field_alloc();
        zfp_field_set_pointer(field, decompressed);

        zfp = zfp_stream_open(NULL);
        stream = stream_open((void *)compressedBytes.c_str(), bytesSize);
        zfp_stream_set_bit_stream(zfp, stream);

        // DECOMPRESS ALL THE THINGS

        zfp_stream_rewind(zfp);
        zfp_read_header(zfp, field, ZFP_HEADER_FULL);

        if (!zfp_decompress(zfp, field))
            fprintf(stderr, "decompression failed\n");

        // CLEANUP

        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        return (float*)decompressed;
    }
}
