#include "zfp.h"
#include <emscripten.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define CBASE64_IMPLEMENTATION
#include "base64.h"

static cbase64_encodestate encodeState;
static cbase64_decodestate decodeState;

static char encodeBuff[8388608];
static unsigned char decodeBuff[8388608];

// https://github.com/SizzlingCalamari/cbase64/blob/master/tests/encodedecode.c
EncodeData(const unsigned char* data_in, unsigned int length_in, unsigned int* length_out)
{
    char* encodeBuffEnd = encodeBuff;

    cbase64_encodestate encodeState;
    cbase64_init_encodestate(&encodeState);
    encodeBuffEnd += cbase64_encode_block(data_in, length_in, encodeBuffEnd, &encodeState);
    encodeBuffEnd += cbase64_encode_blockend(encodeBuffEnd, &encodeState);

    *length_out = (encodeBuffEnd - encodeBuff);
}

DecodeData(const char* code_in, unsigned int length_in, unsigned int* length_out)
{
    const unsigned int decodedLength = cbase64_calc_decoded_length(code_in, length_in);

    cbase64_decodestate decodeState;
    cbase64_init_decodestate(&decodeState);
    *length_out = cbase64_decode_block(code_in, length_in, decodeBuff, &decodeState);
}

EMSCRIPTEN_KEEPALIVE
char* zfpHelper(const char* data)
{
    const unsigned int dataLength = 1 + strlen(data);

    unsigned int encodedLength;
    unsigned int decodedLength;

    DecodeData(encodeBuff, encodedLength, &decodedLength);
    // TODO
    EncodeData((const unsigned char*)decodeBuff, decodedLength, &encodedLength);

    return encodeBuff;
}
