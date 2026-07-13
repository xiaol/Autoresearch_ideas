import { LensTokenMessage } from '@/lib/utils/lens';
import { describe, expect, it } from 'vitest';
import { detectChatFormat, groupTokensIntoMessages } from '../jlens-chat-format';

function token(position: number, value: string): LensTokenMessage {
  return {
    kind: 'token',
    position,
    token: value,
    id: position + 1,
    is_generated: false,
    results: [],
  };
}

describe('RWKV JLens chat format', () => {
  it('detects RWKV before the ChatML fallback', () => {
    expect(detectChatFormat('rwkv7-g1d-0-1b').id).toBe('rwkv');
  });

  it('groups split plain-text role headers and keeps the turn separator in the footer', () => {
    const values = [
      'User',
      ':',
      ' Hello',
      '\n\n',
      'Ass',
      'istant',
      ':',
      ' <think',
      '>',
      '\n',
      '</',
      'think',
      '>',
      ' Paris',
      '.',
    ];
    const tokens = values.map((value, index) => token(index, value));
    const grouped = groupTokensIntoMessages(tokens, detectChatFormat('rwkv7-g1d-0-1b'));

    expect(grouped.hasChatFormat).toBe(true);
    expect(grouped.messages).toHaveLength(2);
    expect(grouped.messages.map((message) => message.role)).toEqual(['user', 'assistant']);
    expect(grouped.messages[0].headerTokens.map((item) => item.token).join('')).toBe('User:');
    expect(grouped.messages[0].contentTokens.map((item) => item.token).join('')).toBe(' Hello');
    expect(grouped.messages[0].footerTokens.map((item) => item.token).join('')).toBe('\n\n');
    expect(grouped.messages[1].headerTokens.map((item) => item.token).join('')).toBe('Assistant:');
    expect(grouped.messages[1].contentTokens.map((item) => item.token).join('')).toContain(' Paris.');
  });
});
