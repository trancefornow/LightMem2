import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { reserveUnusedPort } from "@lightmem2/host-adapter";

import { normalizeTokenPilotCodexConfig } from "./config.js";
import {
  buildCodexEffectiveHistory,
  loadCodexContextHistoryJournal,
  type JsonObject,
} from "./context-history/index.js";
import type { TokenPilotCodexLogger } from "./logger.js";
import { startCodexResponsesProxy, type CodexProxyRuntime } from "./proxy-runtime.js";
import { readLatestCodexRebaseEpoch } from "./context-rewrite/index.js";
import type { CodexRebaseAccounting } from "./context-rewrite/types.js";
import { resolveCodexSessionIdByResponseId } from "./session-state.js";

export const CODEX_REBASE_PROVIDER_SMOKE_EVIDENCE_SCHEMA =
  "lightmem2.codex.context-rebase-provider-smoke-evidence/v1";

export type ProviderUsageObservation = {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  totalTokens: number;
};

export type ProviderUsageTurnComparison = {
  turn: number;
  baselineInputTokens: number;
  rebaseInputTokens: number;
  savedInputTokens: number;
  cumulativeSavedInputTokens: number;
  baselineCachedInputTokens: number;
  rebaseCachedInputTokens: number;
};

export type ProviderUsageEvidence = {
  source: "provider-response-usage";
  setup: {
    baseline: ProviderUsageObservation[];
    rebase: ProviderUsageObservation[];
  };
  continuationTurns: ProviderUsageTurnComparison[];
  observedBreakEvenTurn?: number;
  projectedBreakEvenTurn?: number;
  rebaseTurnOverheadTokens: number;
  observedSavedInputTokens: number;
  subsequentSavedInputTokensPerTurn: number;
};

export type CodexRebaseProviderSmokeEvidence = {
  schema: typeof CODEX_REBASE_PROVIDER_SMOKE_EVIDENCE_SCHEMA;
  mode: "provider";
  provider: "openai-compatible";
  endpointHost: string;
  model: string;
  runtime: {
    node: string;
    codexCli: string;
  };
  startedAt: string;
  finishedAt: string;
  capability: {
    responsesEndpointAccepted: boolean;
    reasoningItemPresent: boolean;
    encryptedReasoningPresent: boolean;
    encryptedPayloadChars: number;
    encryptedPayloadSha256: string;
    toolCallPresent: boolean;
  };
  rebase: {
    committed: boolean;
    oldChainReferenceRemoved: boolean;
    sentinel: {
      evictedAbsent: boolean;
      retainedPresent: boolean;
    };
    replayItemTypes: string[];
    encryptedPayloadDigestMatches: boolean;
    toolClosure: {
      callCount: number;
      outputCount: number;
      complete: boolean;
    };
    responseChain: {
      newRootPresent: boolean;
      terminalPresent: boolean;
      terminalSessionMappingMatches: boolean;
      journalCommittedBeforeEpoch: boolean;
      continuationTurns: number;
      linksValid: boolean;
      restartPreserved: boolean;
      finalHistoryComplete: boolean;
    };
    estimatorAccounting?: CodexRebaseAccounting;
  };
  usage: ProviderUsageEvidence;
  privacy: {
    credentialSource: "OPENAI_API_KEY";
    baseUrlSource: "OPENAI_BASE_URL-or-cli";
    rawPromptPersisted: false;
    rawEncryptedPayloadPersisted: false;
    rawResponseIdPersisted: false;
    rawHeadersPersisted: false;
    ephemeralStateRemoved: true;
  };
};

export type RunCodexRebaseProviderSmokeOptions = {
  baseUrl: string;
  model?: string;
  outputDir?: string;
  continuationTurns?: number;
};

export type CodexRebaseProviderSmokeRunResult = {
  artifactPath: string;
  artifactSha256: string;
  evidence: CodexRebaseProviderSmokeEvidence;
};

type ProviderConversationResult = {
  setupUsage: ProviderUsageObservation[];
  continuationUsage: ProviderUsageObservation[];
};

type ProviderRebaseResult = ProviderConversationResult & {
  capability: CodexRebaseProviderSmokeEvidence["capability"];
  rebase: CodexRebaseProviderSmokeEvidence["rebase"];
};

const FETCH_FORBIDDEN_PORTS = new Set([
  1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77,
  79, 87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123,
  135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514, 515, 526, 530,
  531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990, 993, 995, 1719,
  1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666,
  6667, 6668, 6669, 6697, 10080,
]);

const silentLogger: TokenPilotCodexLogger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function jsonItems(value: unknown): JsonObject[] {
  return Array.isArray(value)
    ? value.filter((item): item is JsonObject => Boolean(item && typeof item === "object" && !Array.isArray(item)))
    : [];
}

function numericField(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : 0;
}

function usageObservation(response: JsonObject): ProviderUsageObservation {
  const usage = response.usage && typeof response.usage === "object" && !Array.isArray(response.usage)
    ? response.usage as JsonObject
    : {};
  const details = usage.input_tokens_details
    && typeof usage.input_tokens_details === "object"
    && !Array.isArray(usage.input_tokens_details)
    ? usage.input_tokens_details as JsonObject
    : {};
  return {
    inputTokens: numericField(usage.input_tokens),
    cachedInputTokens: numericField(details.cached_tokens),
    outputTokens: numericField(usage.output_tokens),
    totalTokens: numericField(usage.total_tokens),
  };
}

function responseId(response: JsonObject, phase: string): string {
  if (typeof response.id !== "string" || !response.id) {
    throw new Error(`Provider smoke ${phase} did not return a response id`);
  }
  if (response.status && response.status !== "completed") {
    throw new Error(`Provider smoke ${phase} returned a non-completed response`);
  }
  return response.id;
}

function toolClosureEvidence(input: JsonObject[]): {
  callCount: number;
  outputCount: number;
  complete: boolean;
} {
  const calls = input.filter((item) => item.type === "function_call");
  const outputs = input.filter((item) => item.type === "function_call_output");
  const callIds = calls.map((item) => item.call_id).filter((value): value is string => typeof value === "string");
  const outputIds = outputs
    .map((item) => item.call_id)
    .filter((value): value is string => typeof value === "string");
  const complete = callIds.length === calls.length
    && outputIds.length === outputs.length
    && new Set(callIds).size === callIds.length
    && new Set(outputIds).size === outputIds.length
    && callIds.length === outputIds.length
    && callIds.every((callId) => outputIds.includes(callId));
  return { callCount: calls.length, outputCount: outputs.length, complete };
}

async function reserveFetchPort(): Promise<number> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const port = await reserveUnusedPort();
    if (!FETCH_FORBIDDEN_PORTS.has(port)) return port;
  }
  throw new Error("Unable to reserve a fetch-safe provider smoke port");
}

function buildProviderSmokeConfig(params: {
  stateDir: string;
  proxyPort: number;
  upstreamBaseUrl: string;
  rewriteEnabled: boolean;
}) {
  return normalizeTokenPilotCodexConfig({
    stateDir: params.stateDir,
    proxyPort: params.proxyPort,
    upstreamProvider: "provider-smoke",
    upstream: {
      name: "openai-compatible",
      baseUrl: params.upstreamBaseUrl,
      wireApi: "responses",
      requiresOpenAIAuth: true,
    },
    modules: {
      stabilizer: false,
      reduction: false,
    },
    contextRewrite: {
      enabled: params.rewriteEnabled,
      mode: "response_chain_rebase",
      failureMode: "bypass",
      retryOriginalRequest: true,
      cooldownMs: 300_000,
      mutationPlan: { operations: [] },
    },
  });
}

async function postProviderResponse(
  runtime: CodexProxyRuntime,
  payload: JsonObject,
  phase: string,
): Promise<JsonObject> {
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    let response: Response;
    try {
      response = await fetch(`${runtime.baseUrl}/responses`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (error) {
      const kind = error instanceof Error ? error.name : "unknown";
      throw new Error(`Provider smoke ${phase} failed before receiving HTTP response (${kind})`);
    }
    const text = await response.text();
    if (!response.ok) {
      let code = "unknown";
      let category = "unclassified";
      try {
        const parsed = JSON.parse(text) as { error?: { code?: unknown; type?: unknown; message?: unknown } };
        const rawCode = parsed.error?.code ?? parsed.error?.type;
        if (typeof rawCode === "string" && /^[a-zA-Z0-9_.-]{1,80}$/.test(rawCode)) code = rawCode;
        const message = typeof parsed.error?.message === "string" ? parsed.error.message : "";
        if (/previous_response_id/i.test(message)) category = "chain-reference";
        else if (/tool_choice/i.test(message)) category = "tool-choice";
        else if (/\btools?\b/i.test(message)) category = "tools";
        else if (/\bstore\b/i.test(message)) category = "storage";
        else if (/reasoning/i.test(message)) category = "reasoning";
        else if (/encrypted_content/i.test(message)) category = "encrypted-replay";
        else if (/\binput\b/i.test(message)) category = "input-schema";
      } catch {
        // Do not persist or print raw provider error bodies.
      }
      const transient = response.status === 429 || response.status >= 500;
      if (transient && attempt < 3) {
        await new Promise((resolveDelay) => setTimeout(resolveDelay, attempt * 500));
        continue;
      }
      throw new Error(`Provider smoke ${phase} failed with HTTP ${response.status} (${code}; ${category})`);
    }
    let parsed: JsonObject;
    try {
      parsed = JSON.parse(text) as JsonObject;
    } catch {
      throw new Error(`Provider smoke ${phase} returned malformed JSON`);
    }
    responseId(parsed, phase);
    return parsed;
  }
  throw new Error(`Provider smoke ${phase} exhausted its transient retry budget`);
}

function toolDefinition(): JsonObject {
  return {
    type: "function",
    name: "lookup_smoke_fixture",
    description: "Return one fixed synthetic smoke-test record.",
    strict: true,
    parameters: {
      type: "object",
      properties: {
        record: { type: "string" },
      },
      required: ["record"],
      additionalProperties: false,
    },
  };
}

function firstTurnPayload(params: {
  model: string;
  sessionId: string;
  evictText: string;
  keepText: string;
}): JsonObject {
  return {
    model: params.model,
    stream: false,
    store: false,
    include: ["reasoning.encrypted_content"],
    reasoning: { effort: "medium", summary: "auto" },
    max_output_tokens: 512,
    metadata: { tokenpilotSessionId: params.sessionId },
    instructions: "Reason about the two synthetic records, then reply with the single word READY.",
    input: [
      { role: "user", content: params.evictText },
      { role: "user", content: params.keepText },
    ],
  };
}

function continuationPayload(params: {
  model: string;
  previousResponseId: string;
  input: JsonObject[];
}): JsonObject {
  return {
    model: params.model,
    stream: false,
    store: true,
    include: ["reasoning.encrypted_content"],
    reasoning: { effort: "medium", summary: "auto" },
    max_output_tokens: 512,
    tools: [toolDefinition()],
    tool_choice: "none",
    previous_response_id: params.previousResponseId,
    input: params.input,
  };
}

function toolRequestItem(): JsonObject {
  return { role: "user", content: "Use lookup_smoke_fixture for the retained synthetic record." };
}

function requiredToolCallPayload(params: {
  model: string;
  sessionId: string;
  historyInput: JsonObject[];
}): JsonObject {
  return {
    model: params.model,
    stream: false,
    store: false,
    include: ["reasoning.encrypted_content"],
    reasoning: { effort: "medium", summary: "auto" },
    max_output_tokens: 512,
    metadata: { tokenpilotSessionId: params.sessionId },
    tools: [toolDefinition()],
    tool_choice: { type: "function", name: "lookup_smoke_fixture" },
    input: [...params.historyInput, toolRequestItem()],
  };
}

function storedRootPayload(params: {
  model: string;
  sessionId: string;
  historyInput: JsonObject[];
}): JsonObject {
  return {
    model: params.model,
    stream: false,
    store: true,
    include: ["reasoning.encrypted_content"],
    reasoning: { effort: "medium", summary: "auto" },
    max_output_tokens: 512,
    metadata: { tokenpilotSessionId: params.sessionId },
    tools: [toolDefinition()],
    tool_choice: "none",
    input: params.historyInput,
  };
}

function firstReasoning(response: JsonObject): {
  reasoning: JsonObject;
  encryptedPayload: string;
} {
  const output = jsonItems(response.output);
  const reasoning = output.find((item) => item.type === "reasoning");
  const encryptedPayload = typeof reasoning?.encrypted_content === "string"
    ? reasoning.encrypted_content
    : "";
  if (!reasoning) {
    const outputTypes = output
      .map((item) => typeof item.type === "string" ? item.type : "unknown")
      .join(",") || "none";
    throw new Error(`Provider response did not include a reasoning item (output types: ${outputTypes})`);
  }
  if (!encryptedPayload) throw new Error("Provider response did not include encrypted reasoning content");
  return { reasoning, encryptedPayload };
}

function requiredToolCall(response: JsonObject): { call: JsonObject; callId: string } {
  const output = jsonItems(response.output);
  const call = output.find((item) => item.type === "function_call" && item.name === "lookup_smoke_fixture");
  const callId = typeof call?.call_id === "string" ? call.call_id : "";
  if (!call || !callId) throw new Error("Provider response did not include the required function call");
  return { call, callId };
}

async function runControlConversation(params: {
  baseUrl: string;
  model: string;
  continuationTurns: number;
  marker: string;
}): Promise<ProviderConversationResult> {
  const stateDir = await mkdtemp(join(tmpdir(), "lightmem2-codex-provider-control-state-"));
  let runtime: CodexProxyRuntime | undefined;
  try {
    const config = buildProviderSmokeConfig({
      stateDir,
      proxyPort: await reserveFetchPort(),
      upstreamBaseUrl: params.baseUrl,
      rewriteEnabled: false,
    });
    runtime = await startCodexResponsesProxy({ config, logger: silentLogger });
    const sessionId = `codex-provider-control-${randomUUID()}`;
    const evictText = `EVICT_ME_${params.marker}\n${"discardable provider smoke context. ".repeat(180)}`;
    const keepText = `KEEP_ME_${params.marker}`;
    const first = await postProviderResponse(runtime, firstTurnPayload({
      model: params.model,
      sessionId,
      evictText,
      keepText,
    }), "control setup turn 1");
    firstReasoning(first);
    const initialHistory = [
      { role: "user", content: evictText },
      { role: "user", content: keepText },
      ...jsonItems(first.output),
    ];
    const toolResponse = await postProviderResponse(runtime, requiredToolCallPayload({
      model: params.model,
      sessionId,
      historyInput: initialHistory,
    }), "control setup turn 2");
    const toolCall = requiredToolCall(toolResponse);
    const toolOutputResponse = await postProviderResponse(runtime, storedRootPayload({
      model: params.model,
      sessionId,
      historyInput: [
        ...initialHistory,
        toolRequestItem(),
        ...jsonItems(toolResponse.output),
        {
          type: "function_call_output",
          call_id: toolCall.callId,
          output: "retained synthetic tool result",
        },
      ],
    }), "control setup turn 3");
    let previousResponseId = responseId(toolOutputResponse, "control setup turn 3");
    const continuationUsage: ProviderUsageObservation[] = [];
    for (let turn = 1; turn <= params.continuationTurns; turn += 1) {
      const response = await postProviderResponse(runtime, continuationPayload({
        model: params.model,
        previousResponseId,
        input: [{ role: "user", content: `Acknowledge continuation turn ${turn} with one word.` }],
      }), `control continuation turn ${turn}`);
      previousResponseId = responseId(response, `control continuation turn ${turn}`);
      continuationUsage.push(usageObservation(response));
    }
    return {
      setupUsage: [
        usageObservation(first),
        usageObservation(toolResponse),
        usageObservation(toolOutputResponse),
      ],
      continuationUsage,
    };
  } finally {
    await runtime?.close();
    await rm(stateDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 20 });
  }
}

async function runRebaseConversation(params: {
  baseUrl: string;
  model: string;
  continuationTurns: number;
  marker: string;
}): Promise<ProviderRebaseResult> {
  const stateDir = await mkdtemp(join(tmpdir(), "lightmem2-codex-provider-rebase-state-"));
  const sessionId = `codex-provider-rebase-${randomUUID()}`;
  const evictText = `EVICT_ME_${params.marker}\n${"discardable provider smoke context. ".repeat(180)}`;
  const keepText = `KEEP_ME_${params.marker}`;
  let runtime: CodexProxyRuntime | undefined;
  try {
    const config = buildProviderSmokeConfig({
      stateDir,
      proxyPort: await reserveFetchPort(),
      upstreamBaseUrl: params.baseUrl,
      rewriteEnabled: true,
    });
    runtime = await startCodexResponsesProxy({ config, logger: silentLogger });
    const first = await postProviderResponse(runtime, firstTurnPayload({
      model: params.model,
      sessionId,
      evictText,
      keepText,
    }), "rebase setup turn 1");
    const firstOutput = firstReasoning(first);
    const initialHistory = [
      { role: "user", content: evictText },
      { role: "user", content: keepText },
      ...jsonItems(first.output),
    ];
    const toolResponse = await postProviderResponse(runtime, requiredToolCallPayload({
      model: params.model,
      sessionId,
      historyInput: initialHistory,
    }), "rebase setup turn 2");
    const toolCall = requiredToolCall(toolResponse);
    const toolOutputResponse = await postProviderResponse(runtime, storedRootPayload({
      model: params.model,
      sessionId,
      historyInput: [
        ...initialHistory,
        toolRequestItem(),
        ...jsonItems(toolResponse.output),
        {
          type: "function_call_output",
          call_id: toolCall.callId,
          output: "retained synthetic tool result",
        },
      ],
    }), "rebase setup turn 3");
    let previousResponseId = responseId(toolOutputResponse, "rebase setup turn 3");
    const beforeRebase = await buildCodexEffectiveHistory({
      stateDir,
      sessionId,
      headResponseId: previousResponseId,
    });
    const evictedItem = beforeRebase.replayableItems.find((entry) => (
      JSON.stringify(entry.item).includes(`EVICT_ME_${params.marker}`)
    ));
    if (!evictedItem) throw new Error("Provider smoke could not resolve the eviction target");
    config.contextRewrite.mutationPlan = {
      operations: [{ type: "evict", stableItemId: evictedItem.stableItemId }],
    };

    const continuationUsage: ProviderUsageObservation[] = [];
    const rebaseResponse = await postProviderResponse(runtime, continuationPayload({
      model: params.model,
      previousResponseId,
      input: [{ role: "user", content: "Acknowledge continuation turn 1 with one word." }],
    }), "rebase continuation turn 1");
    previousResponseId = responseId(rebaseResponse, "rebase continuation turn 1");
    continuationUsage.push(usageObservation(rebaseResponse));
    config.contextRewrite.mutationPlan = { operations: [] };

    let restartPreserved = false;
    const expectedPreviousIds: string[] = [];
    for (let turn = 2; turn <= params.continuationTurns; turn += 1) {
      if (turn === 3) {
        const headBeforeRestart = previousResponseId;
        await runtime.close();
        runtime = undefined;
        const restartedConfig = buildProviderSmokeConfig({
          stateDir,
          proxyPort: await reserveFetchPort(),
          upstreamBaseUrl: params.baseUrl,
          rewriteEnabled: true,
        });
        runtime = await startCodexResponsesProxy({ config: restartedConfig, logger: silentLogger });
        restartPreserved = await resolveCodexSessionIdByResponseId(stateDir, headBeforeRestart) === sessionId;
      }
      expectedPreviousIds.push(previousResponseId);
      const response = await postProviderResponse(runtime, continuationPayload({
        model: params.model,
        previousResponseId,
        input: [{ role: "user", content: `Acknowledge continuation turn ${turn} with one word.` }],
      }), `rebase continuation turn ${turn}`);
      previousResponseId = responseId(response, `rebase continuation turn ${turn}`);
      continuationUsage.push(usageObservation(response));
    }

    const epoch = await readLatestCodexRebaseEpoch({ stateDir, sessionId });
    const journal = await loadCodexContextHistoryJournal(stateDir, sessionId);
    const committedRootResponse = journal.find((entry) => (
      entry.kind === "response"
      && entry.responseId === responseId(rebaseResponse, "rebase continuation turn 1")
      && entry.status === "completed"
      && entry.previousResponseId === null
    ));
    const committedRootRequest = committedRootResponse?.requestId
      ? journal.find((entry) => (
        entry.kind === "request"
        && entry.requestId === committedRootResponse.requestId
        && entry.status === "completed"
        && Array.isArray(entry.committedInputItems)
      ))
      : undefined;
    const replayInput = committedRootRequest?.kind === "request"
      ? jsonItems(committedRootRequest.committedInputItems)
      : [];
    const replayText = JSON.stringify(replayInput);
    const replayReasoning = replayInput.filter((item) => item.type === "reasoning");
    const digestMatches = replayReasoning.some((item) => (
      typeof item.encrypted_content === "string"
      && sha256(item.encrypted_content) === sha256(firstOutput.encryptedPayload)
    ));
    const finalHistory = await buildCodexEffectiveHistory({
      stateDir,
      sessionId,
      headResponseId: previousResponseId,
    });
    const finalHistoryText = JSON.stringify(finalHistory.replayableItems);
    const continuationRequestLinks = expectedPreviousIds.map((expected) => journal.some((entry) => (
      entry.kind === "request"
      && entry.status === "completed"
      && entry.previousResponseId === expected
    )));
    const terminalSessionMappingMatches =
      await resolveCodexSessionIdByResponseId(stateDir, previousResponseId) === sessionId;
    const encryptedDigest = sha256(firstOutput.encryptedPayload);

    return {
      setupUsage: [
        usageObservation(first),
        usageObservation(toolResponse),
        usageObservation(toolOutputResponse),
      ],
      continuationUsage,
      capability: {
        responsesEndpointAccepted: true,
        reasoningItemPresent: true,
        encryptedReasoningPresent: true,
        encryptedPayloadChars: firstOutput.encryptedPayload.length,
        encryptedPayloadSha256: encryptedDigest,
        toolCallPresent: true,
      },
      rebase: {
        committed: epoch?.status === "committed",
        oldChainReferenceRemoved: Boolean(committedRootResponse),
        sentinel: {
          evictedAbsent: !replayText.includes(`EVICT_ME_${params.marker}`)
            && !finalHistoryText.includes(`EVICT_ME_${params.marker}`),
          retainedPresent: replayText.includes(`KEEP_ME_${params.marker}`)
            && finalHistoryText.includes(`KEEP_ME_${params.marker}`),
        },
        replayItemTypes: replayInput.map((item) => (
          typeof item.type === "string"
            ? item.type
            : typeof item.role === "string" ? `message:${item.role}` : "unknown"
        )),
        encryptedPayloadDigestMatches: digestMatches,
        toolClosure: toolClosureEvidence(replayInput),
        responseChain: {
          newRootPresent: responseId(rebaseResponse, "rebase continuation turn 1").length > 0,
          terminalPresent: previousResponseId.length > 0,
          terminalSessionMappingMatches,
          journalCommittedBeforeEpoch:
            epoch?.status === "committed"
            && epoch.newResponseId === responseId(rebaseResponse, "rebase continuation turn 1")
            && Boolean(committedRootResponse)
            && Boolean(committedRootRequest)
            && Date.parse(committedRootResponse?.observedAt ?? "") <= Date.parse(epoch.updatedAt),
          continuationTurns: params.continuationTurns,
          linksValid: continuationRequestLinks.every(Boolean),
          restartPreserved,
          finalHistoryComplete: !finalHistory.incomplete,
        },
        estimatorAccounting: epoch?.accounting ? { ...epoch.accounting } : undefined,
      },
    };
  } finally {
    await runtime?.close();
    await rm(stateDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 20 });
  }
}

export function compareProviderUsage(
  baseline: ProviderUsageObservation[],
  rebase: ProviderUsageObservation[],
  setup: { baseline: ProviderUsageObservation[]; rebase: ProviderUsageObservation[] },
): ProviderUsageEvidence {
  if (baseline.length !== rebase.length || baseline.length === 0) {
    throw new Error("Provider smoke usage comparison requires equal non-empty turn sets");
  }
  let cumulativeSavedInputTokens = 0;
  const continuationTurns = baseline.map((baselineUsage, index) => {
    const rebaseUsage = rebase[index];
    const savedInputTokens = baselineUsage.inputTokens - rebaseUsage.inputTokens;
    cumulativeSavedInputTokens += savedInputTokens;
    return {
      turn: index + 1,
      baselineInputTokens: baselineUsage.inputTokens,
      rebaseInputTokens: rebaseUsage.inputTokens,
      savedInputTokens,
      cumulativeSavedInputTokens,
      baselineCachedInputTokens: baselineUsage.cachedInputTokens,
      rebaseCachedInputTokens: rebaseUsage.cachedInputTokens,
    };
  });
  const observedBreakEvenTurn = continuationTurns.find((turn) => turn.cumulativeSavedInputTokens >= 0)?.turn;
  const rebaseTurnOverheadTokens = Math.max(
    0,
    rebase[0].inputTokens - baseline[0].inputTokens,
  );
  const subsequentSavings = continuationTurns.slice(1).map((turn) => turn.savedInputTokens);
  const subsequentSavedInputTokensPerTurn = subsequentSavings.length > 0
    ? Math.round(subsequentSavings.reduce((sum, value) => sum + value, 0) / subsequentSavings.length)
    : continuationTurns[0].savedInputTokens;
  const projectedBreakEvenTurn = observedBreakEvenTurn
    ?? (subsequentSavedInputTokensPerTurn > 0
      ? 1 + Math.ceil(rebaseTurnOverheadTokens / subsequentSavedInputTokensPerTurn)
      : undefined);
  return {
    source: "provider-response-usage",
    setup,
    continuationTurns,
    observedBreakEvenTurn,
    projectedBreakEvenTurn,
    rebaseTurnOverheadTokens,
    observedSavedInputTokens: cumulativeSavedInputTokens,
    subsequentSavedInputTokensPerTurn,
  };
}

async function writeEvidence(
  outputDir: string,
  evidence: CodexRebaseProviderSmokeEvidence,
): Promise<{ artifactPath: string; artifactSha256: string }> {
  await mkdir(outputDir, { recursive: true });
  const artifactPath = join(outputDir, "codex-context-rebase-provider-smoke.json");
  const tempPath = `${artifactPath}.${process.pid}.${randomUUID()}.tmp`;
  const text = `${JSON.stringify(evidence, null, 2)}\n`;
  await writeFile(tempPath, text, { encoding: "utf8", mode: 0o600 });
  await rename(tempPath, artifactPath);
  return { artifactPath, artifactSha256: sha256(text) };
}

export async function runCodexRebaseProviderSmoke(
  options: RunCodexRebaseProviderSmokeOptions,
): Promise<CodexRebaseProviderSmokeRunResult> {
  if (!process.env.OPENAI_API_KEY?.trim()) {
    throw new Error("Provider smoke requires OPENAI_API_KEY in the environment");
  }
  const baseUrl = options.baseUrl.trim().replace(/\/+$/, "");
  let parsedBaseUrl: URL;
  try {
    parsedBaseUrl = new URL(baseUrl);
  } catch {
    throw new Error("Provider smoke requires a valid base URL");
  }
  if (parsedBaseUrl.protocol !== "https:" && parsedBaseUrl.hostname !== "127.0.0.1" && parsedBaseUrl.hostname !== "localhost") {
    throw new Error("Provider smoke requires HTTPS unless the provider is loopback-only");
  }
  const model = options.model?.trim() || "gpt-5.4-mini";
  const continuationTurns = options.continuationTurns ?? 5;
  if (!Number.isInteger(continuationTurns) || continuationTurns < 2 || continuationTurns > 20) {
    throw new Error("Provider smoke continuationTurns must be an integer from 2 to 20");
  }
  const startedAt = new Date().toISOString();
  const marker = randomUUID();
  // Proxy startup configures a process-global resolver, so the scenarios run serially.
  const baseline = await runControlConversation({
    baseUrl,
    model,
    continuationTurns,
    marker,
  });
  const rebase = await runRebaseConversation({
    baseUrl,
    model,
    continuationTurns,
    marker,
  });
  const usage = compareProviderUsage(
    baseline.continuationUsage,
    rebase.continuationUsage,
    { baseline: baseline.setupUsage, rebase: rebase.setupUsage },
  );
  const evidence: CodexRebaseProviderSmokeEvidence = {
    schema: CODEX_REBASE_PROVIDER_SMOKE_EVIDENCE_SCHEMA,
    mode: "provider",
    provider: "openai-compatible",
    endpointHost: parsedBaseUrl.hostname,
    model,
    runtime: {
      node: process.version,
      codexCli: process.env.CODEX_CLI_VERSION?.trim() || "not-observed",
    },
    startedAt,
    finishedAt: new Date().toISOString(),
    capability: rebase.capability,
    rebase: rebase.rebase,
    usage,
    privacy: {
      credentialSource: "OPENAI_API_KEY",
      baseUrlSource: "OPENAI_BASE_URL-or-cli",
      rawPromptPersisted: false,
      rawEncryptedPayloadPersisted: false,
      rawResponseIdPersisted: false,
      rawHeadersPersisted: false,
      ephemeralStateRemoved: true,
    },
  };
  const outputDir = options.outputDir
    ? resolve(options.outputDir)
    : await mkdtemp(join(tmpdir(), "lightmem2-codex-provider-smoke-evidence-"));
  const artifact = await writeEvidence(outputDir, evidence);
  return { ...artifact, evidence };
}
