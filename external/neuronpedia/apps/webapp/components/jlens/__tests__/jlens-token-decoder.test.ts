import { LensPromptToken, LensTokenMessage } from '@/lib/utils/lens';
import { describe, expect, it } from 'vitest';
import { LensTokenDisplayDecoder } from '../jlens-token-decoder';

function promptToken(position: number, token: string, tokenBytes?: number[]): LensPromptToken {
  return { position, token, token_bytes: tokenBytes, id: position + 1, is_generated: false };
}

function streamedToken(position: number, token: string, tokenBytes?: number[]): LensTokenMessage {
  return {
    kind: 'token',
    position,
    token,
    token_bytes: tokenBytes,
    id: position + 1,
    is_generated: true,
    results: [],
  };
}

describe('LensTokenDisplayDecoder', () => {
  it('incrementally decodes a code point split across RWKV generated tokens', () => {
    const decoder = new LensTokenDisplayDecoder('rwkv7-g1d-0-1b');
    const fragments = [
      streamedToken(0, '\ufffd', [0xf0, 0x9f]),
      streamedToken(1, '\ufffd', [0x98]),
      streamedToken(2, '\ufffd', [0x80]),
      streamedToken(3, '!', [0x21]),
    ].map((token) => decoder.normalizeToken(token));

    expect(fragments.map((token) => token.token)).toEqual(['', '', '😀', '!']);
    expect(fragments.map((token) => token.token).join('')).toBe('😀!');
    expect(decoder.normalizeCompletion('\ufffd\ufffd\ufffd!')).toBe('😀!');
  });

  it('uses the same decoded prompt text when readouts replace prompt records', () => {
    const decoder = new LensTokenDisplayDecoder('rwkv7-g1d-0-1b');
    const prompt = decoder.normalizePromptTokens([
      promptToken(0, 'Hi ', [0x48, 0x69, 0x20]),
      promptToken(1, '\ufffd', [0xe4, 0xbd]),
      promptToken(2, '\ufffd', [0xa0]),
    ]);
    const readout = decoder.normalizeToken({
      ...streamedToken(1, '\ufffd', [0xe4, 0xbd]),
      is_generated: false,
    });

    expect(prompt.map((token) => token.token)).toEqual(['Hi ', '', '你']);
    expect(readout.token).toBe('');
  });

  it('leaves non-RWKV token strings unchanged', () => {
    const decoder = new LensTokenDisplayDecoder('google/gemma-3-1b-it');
    const token = streamedToken(0, 'server text', [0xf0, 0x9f, 0x98, 0x80]);

    expect(decoder.normalizeToken(token)).toBe(token);
    expect(decoder.normalizeCompletion('server completion')).toBe('server completion');
  });
});
