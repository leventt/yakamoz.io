#include <string>
#include <emscripten.h>
#include <zfp.h>

static zfp_type type;     /* array scalar type */
static zfp_field* field;  /* array meta data */
static zfp_stream* zfp;   /* compressed stream */
static bitstream* stream; /* bit stream to write to or read from */

extern "C"
{
    EMSCRIPTEN_KEEPALIVE
    void* zfpHelper(unsigned char* compressedBytes, unsigned int bytesSize)
    {
        // HEAP STUFF
        void* zfpBuffer = malloc(120 * 8320 * 3 * 4);

        // ZPF STUFF

        type = zfp_type_float;
        // math.ceil(4 * 29.97), 8320, 3 (framecount, vert count, dimension count)
        field = zfp_field_3d(zfpBuffer, type, 120, 8320, 3);
        zfp = zfp_stream_open(NULL);
        stream = stream_open((void *)zfpBuffer, bytesSize);
        zfp_stream_set_bit_stream(zfp, stream);
        // tolerance 0.001 matching server side python script called main.py
        zfp_stream_set_accuracy(zfp, 0.0001);

        // DECOMPRESS ALL THE THINGS

        zfp_stream_flush(zfp);
        zfp_stream_rewind(zfp);

        FILE *fp = fmemopen((void*)compressedBytes, bytesSize, "r");
        fread(zfpBuffer, 1, bytesSize, fp);
        if (!zfp_decompress(zfp, field))
            fprintf(stderr, "decompression failed\n");

        // CLEANUP

        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);
        // js side frees zfpBuffer!

        return (void*)zfpBuffer;
    }
}
