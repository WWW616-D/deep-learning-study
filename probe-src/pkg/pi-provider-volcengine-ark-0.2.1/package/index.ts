/**
 * Volcengine Ark (火山方舟, OpenAI-compatible coding endpoint) provider for pi.
 *
 * Endpoint: https://ark.cn-beijing.volces.com/api/coding/v3
 * Auth:     API key via $ARK_API_KEY (Bearer token)
 *
 * Model parameters mirror the Alibaba Bailian entry (context window / output /
 * pricing / reasoning behavior are the same hosted models, per request). The
 * only differences are the endpoint, the env var, the lower-case model IDs
 * (and kimi-k2.6 / minimax-m* use plain IDs without a lab prefix on Ark).
 * maxTokens per model verified against the endpoint (a too-large max_tokens
 * returns 400 InvalidParameter naming the exact cap):
 *   glm-5.3 / glm-5.2 128000, deepseek-v4-* 393216, minimax-m* 131072,
 *   kimi-k2.7-code / kimi-k2.6 32768.
 *
 * IMPORTANT: pi's extension config form (pi.registerProvider(name, {models}))
 * does NOT merge provider-level `compat` into each model — only per-model
 * `compat` is honored. So the shared compat fields live in BASE_COMPAT below and
 * are spread into every model's `compat`.
 *
 * Thinking:
 *   - kimi toggle models        -> thinkingFormat "qwen" (enable_thinking), high only
 *   - deepseek-v4 effort models -> thinkingFormat "deepseek" + reasoning_effort, high only
 *   - glm-5.3                   -> plain reasoning_effort low/medium/high/max (verified on
 *                                  Ark: always thinks, cannot be disabled; "low" ~ off)
 *   - glm-5.2 / minimax         -> no thinking param; reasoning_content captured
 *
 * Usage:
 *   export ARK_API_KEY=...
 *   pi            # then /model -> volcengine-ark/<model>
 *   pi --list-models
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE_COMPAT = {
  supportsDeveloperRole: false,
  supportsStore: false,
  supportsStrictMode: false,
  maxTokensField: "max_tokens" as const,
};

const HIGH_ONLY = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: "high",
  xhigh: null,
  max: null,
} as const;

// glm-5.3 on Ark accepts plain reasoning_effort low/medium/high/max (verified).
// Thinking cannot be disabled; "low" produces near-zero reasoning, so off -> "low".
const GLM53_EFFORT = {
  off: "low",
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "max",
  max: "max",
} as const;

export default function (pi: ExtensionAPI) {
  pi.registerProvider("volcengine-ark", {
    name: "Volcengine Ark",
    baseUrl: "https://ark.cn-beijing.volces.com/api/coding/v3",
    apiKey: "$ARK_API_KEY",
    api: "openai-completions",
    models: [
      {
        id: "glm-5.3",
        name: "GLM-5.3",
        reasoning: true,
        input: ["text"],
        cost: { input: 1.1, output: 3.851, cacheRead: 0.275, cacheWrite: 0 },
        contextWindow: 1_000_000,
        maxTokens: 128_000,
        thinkingLevelMap: GLM53_EFFORT,
        compat: { ...BASE_COMPAT, supportsReasoningEffort: true },
      },
      {
        id: "glm-5.2",
        name: "GLM-5.2",
        reasoning: true,
        input: ["text"],
        cost: { input: 1.1, output: 3.851, cacheRead: 0.275, cacheWrite: 0 },
        contextWindow: 1_000_000,
        maxTokens: 128_000,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, supportsReasoningEffort: false },
      },
      {
        id: "kimi-k2.7-code",
        name: "Kimi K2.7 Code",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 0.73, output: 3.5, cacheRead: 0.15, cacheWrite: 0 },
        contextWindow: 262_144,
        maxTokens: 32_768,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, thinkingFormat: "qwen", supportsReasoningEffort: false },
      },
      {
        id: "deepseek-v4-pro",
        name: "DeepSeek V4 Pro",
        reasoning: true,
        input: ["text"],
        cost: { input: 0.435, output: 0.87, cacheRead: 0.003625, cacheWrite: 0 },
        contextWindow: 1_000_000,
        maxTokens: 393_216,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, thinkingFormat: "deepseek", supportsReasoningEffort: true },
      },
      {
        id: "deepseek-v4-flash",
        name: "DeepSeek V4 Flash",
        reasoning: true,
        input: ["text"],
        cost: { input: 0.14, output: 0.28, cacheRead: 0.0028, cacheWrite: 0 },
        contextWindow: 1_000_000,
        maxTokens: 393_216,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, thinkingFormat: "deepseek", supportsReasoningEffort: true },
      },
      {
        id: "minimax-m3",
        name: "MiniMax-M3",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 0.3, output: 1.2, cacheRead: 0.06, cacheWrite: 0 },
        contextWindow: 512_000,
        maxTokens: 131_072,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, supportsReasoningEffort: false },
      },
      {
        id: "minimax-m2.7",
        name: "MiniMax-M2.7",
        reasoning: true,
        input: ["text"],
        cost: { input: 0.3, output: 1.2, cacheRead: 0.06, cacheWrite: 0.375 },
        contextWindow: 204_800,
        maxTokens: 131_072,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, supportsReasoningEffort: false },
      },
      {
        id: "kimi-k2.6",
        name: "Kimi K2.6",
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 0.929, output: 3.858, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 262_144,
        maxTokens: 32_768,
        thinkingLevelMap: HIGH_ONLY,
        compat: { ...BASE_COMPAT, thinkingFormat: "qwen", supportsReasoningEffort: false },
      },
    ],
  });
}
