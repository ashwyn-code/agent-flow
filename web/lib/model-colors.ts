/**
 * A stable color per model, so agents on different models can be told apart
 * on the canvas and matched to the model legend.
 *
 * Known families get fixed hues (Opus purple, Sonnet blue, Haiku teal, GPT
 * green, Gemini amber...), with larger/"pro" tiers brighter than small/"mini"
 * ones. Anything else gets a hue from a hash of its name.
 */
import { formatModelName } from './utils'

const FAMILIES: { pattern: RegExp; color: string }[] = [
  { pattern: /opus/i, color: '#c58bff' },
  { pattern: /sonnet/i, color: '#6fa8ff' },
  { pattern: /haiku/i, color: '#4fd6c8' },
  { pattern: /fable/i, color: '#ff8fc7' },
  { pattern: /gpt-?5.*(mini|nano)|o\d-mini|gpt-?4o-mini|gpt-?4\.1-(mini|nano)/i, color: '#d8f06a' },
  { pattern: /gpt|o\d\b|codex/i, color: '#4fe08c' },
  { pattern: /gemini.*(flash|lite)/i, color: '#ffd166' },
  { pattern: /gemini/i, color: '#ff9f43' },
  { pattern: /llama|mistral|mixtral|qwen|deepseek/i, color: '#ff7a7a' },
]

const FALLBACK = ['#7fd1ff', '#ffb3de', '#b5f27a', '#ffcf7a', '#a9a4ff', '#7affd9', '#ff9e7a', '#d4a5ff']

function hash(text: string): number {
  let h = 2166136261
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/** Color for a model id; undefined when the model isn't known. */
export function modelColor(model: string | undefined): string | undefined {
  if (!model) return undefined
  for (const { pattern, color } of FAMILIES) if (pattern.test(model)) return color
  return FALLBACK[hash(model.toLowerCase()) % FALLBACK.length]
}

/** Short display name, e.g. 'claude-sonnet-4-5-20250929' → 'Sonnet 4.5'. */
export function modelLabel(model: string): string {
  return formatModelName(model)
}
