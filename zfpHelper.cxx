#include <string>
#include <vector>

#include <emscripten.h>
#include <emscripten/bind.h>

#include <zfp.h>

#define CBASE64_IMPLEMENTATION
#include "cbase64.h"

using namespace emscripten;

static float array[120 * 8320 * 3];

static zfp_type type;     /* array scalar type */
static zfp_field* field;  /* array meta data */
static zfp_stream* zfp;   /* compressed stream */
static bitstream* stream; /* bit stream to write to or read from */

// https://github.com/SizzlingCalamari/cbase64/blob/master/tests/encodedecode.c
unsigned char* DecodeData(const char* code_in, unsigned int length_in, unsigned int* length_out)
{
    const unsigned int decodedLength = cbase64_calc_decoded_length(code_in, length_in);
    unsigned char* dataOut = (unsigned char*)malloc(decodedLength);

    cbase64_decodestate decodeState;
    cbase64_init_decodestate(&decodeState);
    *length_out = cbase64_decode_block(code_in, length_in, dataOut, &decodeState);
    return dataOut;
}

extern "C"
{
    EMSCRIPTEN_KEEPALIVE
    float* zfpHelper(std::string base64Bytes)
    {
        // BASE64 STUFF

        unsigned int outSize = cbase64_calc_decoded_length(base64Bytes.c_str(), base64Bytes.size());
        char* compressedBytes = (char*)malloc(outSize + 1);
        unsigned char* decodeDataHandle;
        decodeDataHandle = DecodeData(base64Bytes.c_str(), base64Bytes.size(), &outSize);
        memcpy(compressedBytes, decodeDataHandle, outSize);
        free(decodeDataHandle);
        compressedBytes[outSize] = '\0';

        std::string compressedStr{compressedBytes};
        free(compressedBytes);

        // ZPF STUFF

        type = zfp_type_float;
        // math.ceil(4 * 29.97), 8320, 3 (framecount, vert count, dimension count)
        field = zfp_field_3d(array, type, 120, 8320, 3);
        zfp = zfp_stream_open(NULL);
        stream = stream_open((void *)compressedStr.c_str(), compressedStr.size());
        zfp_stream_set_bit_stream(zfp, stream);
        // tolerance 0.001 matching server side python script called main.py
        zfp_stream_set_accuracy(zfp, 0.001);

        // DECOMPRESS ALL THE THINGS

        zfp_stream_rewind(zfp);
        zfp_decompress(zfp, field);

        // CLEANUP

        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        return &array[0];
    }
}

