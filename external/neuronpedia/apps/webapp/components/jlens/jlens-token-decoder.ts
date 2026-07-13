import { LensPromptToken, LensTokenMessage } from '@/lib/utils/lens';

type ByteBackedToken = Pick<LensPromptToken, 'token' | 'token_bytes'>;

function createUtf8Decoder(): TextDecoder {
  return new TextDecoder('utf-8', { ignoreBOM: true });
}

function toTokenBytes(value: number[] | undefined): Uint8Array | null {
  if (!Array.isArray(value) || value.some((byte) => !Number.isInteger(byte) || byte < 0 || byte > 255)) {
    return null;
  }
  return Uint8Array.from(value);
}

class IncrementalUtf8Decoder {
  private decoder = createUtf8Decoder();

  decode(token: ByteBackedToken): string {
    const bytes = toTokenBytes(token.token_bytes);
    if (bytes !== null) {
      return this.decoder.decode(bytes, { stream: true });
    }

    const pending = this.decoder.decode();
    this.decoder = createUtf8Decoder();
    return pending + token.token;
  }

  finish(): string {
    const pending = this.decoder.decode();
    this.decoder = createUtf8Decoder();
    return pending;
  }
}

/**
 * Reconstructs display text for RWKV's byte-level tokenizer.
 *
 * A completed code point is assigned to the last token that contributes its
 * bytes. Earlier byte fragments display as empty strings, so joining tokens
 * yields the exact decoded text without duplicate glyphs or replacement chars.
 * Other model families, and older backends without `token_bytes`, keep their
 * existing token strings unchanged.
 */
export class LensTokenDisplayDecoder {
  private readonly enabled: boolean;

  private readonly generatedDecoder = new IncrementalUtf8Decoder();

  private readonly promptTextByPosition = new Map<number, string>();

  private generatedText = '';

  private hasGeneratedBytes = false;

  constructor(modelId: string) {
    this.enabled = modelId.toLowerCase().includes('rwkv');
  }

  normalizePromptTokens(tokens: LensPromptToken[]): LensPromptToken[] {
    if (!this.enabled) {
      return tokens;
    }

    const decoder = new IncrementalUtf8Decoder();
    const normalized = tokens.map((token) => {
      const text = decoder.decode(token);
      this.promptTextByPosition.set(token.position, text);
      return text === token.token ? token : { ...token, token: text };
    });
    const trailing = decoder.finish();
    if (trailing && normalized.length > 0) {
      const lastIndex = normalized.length - 1;
      const last = normalized[lastIndex];
      const text = last.token + trailing;
      normalized[lastIndex] = { ...last, token: text };
      this.promptTextByPosition.set(last.position, text);
    }
    return normalized;
  }

  normalizeToken(token: LensTokenMessage): LensTokenMessage {
    if (!this.enabled) {
      return token;
    }

    if (!token.is_generated) {
      const promptText = this.promptTextByPosition.get(token.position);
      return promptText === undefined || promptText === token.token ? token : { ...token, token: promptText };
    }

    if (toTokenBytes(token.token_bytes) !== null) {
      this.hasGeneratedBytes = true;
    }
    const text = this.generatedDecoder.decode(token);
    this.generatedText += text;
    return text === token.token ? token : { ...token, token: text };
  }

  normalizeCompletion(completion: string): string {
    if (!this.enabled || !this.hasGeneratedBytes) {
      return completion;
    }
    this.generatedText += this.generatedDecoder.finish();
    return this.generatedText;
  }
}
