import assert from "node:assert/strict";
import { createServer, type IncomingMessage } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  CODEX_REBASE_PROVIDER_SMOKE_EVIDENCE_SCHEMA,
  compareProviderUsage,
  runCodexRebaseProviderSmoke,
} from "../src/context-rebase-provider-smoke.js";
import type { JsonObject } from "../src/context-history/index.js";

async function requestBody(req: IncomingMessage): Promise<JsonObject> {
  return new Promise<JsonObject>((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk))));
    req.on("error", reject);
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")) as JsonObject);
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function startProviderFixture(): Promise<{
  baseUrl: string;
  authorizationPresent(): boolean;
  close(): Promise<void>;
}> {
  let ordinal = 0;
  let sawAuthorization = true;
  const contextCharsByResponse = new Map<string, number>();
  const server = createServer(async (req, res) => {
    if (req.method !== "POST" || req.url !== "/v1/responses") {
      res.statusCode = 404;
      res.end("not found");
      return;
    }
    sawAuthorization = sawAuthorization && /^Bearer\s+\S+$/.test(String(req.headers.authorization ?? ""));
    const payload = await requestBody(req);
    ordinal += 1;
    const id = `resp-provider-fixture-${ordinal}`;
    const previousId = typeof payload.previous_response_id === "string"
      ? payload.previous_response_id
      : undefined;
    const previousContextChars = previousId ? contextCharsByResponse.get(previousId) ?? 0 : 0;
    const currentChars = JSON.stringify(payload.input ?? []).length;
    const forcedToolCall = payload.tool_choice
      && typeof payload.tool_choice === "object"
      && !Array.isArray(payload.tool_choice);
    const encryptedContent = `opaque-provider-fixture-${ordinal}`;
    const output = forcedToolCall
      ? [
        {
          id: `reasoning-${ordinal}`,
          type: "reasoning",
          encrypted_content: encryptedContent,
          summary: [],
        },
        {
          id: `call-item-${ordinal}`,
          type: "function_call",
          call_id: `call-provider-fixture-${ordinal}`,
          name: "lookup_smoke_fixture",
          arguments: "{\"record\":\"retained\"}",
        },
      ]
      : [
        {
          id: `reasoning-${ordinal}`,
          type: "reasoning",
          encrypted_content: encryptedContent,
          summary: [],
        },
        {
          id: `message-${ordinal}`,
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "OK" }],
        },
      ];
    const outputChars = JSON.stringify(output).length;
    const nextContextChars = previousContextChars + currentChars + outputChars;
    contextCharsByResponse.set(id, nextContextChars);
    const inputTokens = Math.ceil((previousContextChars + currentChars) / 4);
    const outputTokens = Math.ceil(outputChars / 4);
    res.statusCode = 200;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({
      id,
      object: "response",
      status: "completed",
      previous_response_id: previousId,
      output,
      usage: {
        input_tokens: inputTokens,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      },
    }));
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("fixture server did not expose a TCP address");
  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    authorizationPresent: () => sawAuthorization,
    close() {
      return new Promise<void>((resolve, reject) => {
        server.close((error) => error ? reject(error) : resolve());
      });
    },
  };
}

test("provider usage comparison reports observed and projected break-even", () => {
  const observation = (inputTokens: number) => ({
    inputTokens,
    cachedInputTokens: 0,
    outputTokens: 1,
    totalTokens: inputTokens + 1,
  });
  const evidence = compareProviderUsage(
    [observation(120), observation(140), observation(160)],
    [observation(150), observation(100), observation(100)],
    { baseline: [], rebase: [] },
  );
  assert.equal(evidence.observedBreakEvenTurn, 2);
  assert.equal(evidence.projectedBreakEvenTurn, 2);
  assert.equal(evidence.rebaseTurnOverheadTokens, 30);
  assert.equal(evidence.observedSavedInputTokens, 70);
  assert.equal(evidence.subsequentSavedInputTokensPerTurn, 50);
});

test("provider smoke emits sanitized real-chain and usage evidence", async () => {
  const provider = await startProviderFixture();
  const outputDir = await mkdtemp(join(tmpdir(), "lightmem2-codex-provider-smoke-test-"));
  const previousKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "provider-smoke-test-key-not-secret";
  try {
    const result = await runCodexRebaseProviderSmoke({
      baseUrl: provider.baseUrl,
      model: "provider-fixture-model",
      continuationTurns: 3,
      outputDir,
    });
    const evidence = result.evidence;
    const artifactText = await readFile(result.artifactPath, "utf8");

    assert.equal(evidence.schema, CODEX_REBASE_PROVIDER_SMOKE_EVIDENCE_SCHEMA);
    assert.equal(evidence.mode, "provider");
    assert.equal(evidence.capability.encryptedReasoningPresent, true);
    assert.ok(evidence.capability.encryptedPayloadChars > 0);
    assert.equal(evidence.rebase.committed, true);
    assert.equal(evidence.rebase.oldChainReferenceRemoved, true);
    assert.deepEqual(evidence.rebase.sentinel, {
      evictedAbsent: true,
      retainedPresent: true,
    });
    assert.equal(evidence.rebase.encryptedPayloadDigestMatches, true);
    assert.deepEqual(evidence.rebase.toolClosure, {
      callCount: 1,
      outputCount: 1,
      complete: true,
    });
    assert.equal(evidence.rebase.responseChain.continuationTurns, 3);
    assert.equal(evidence.rebase.responseChain.linksValid, true);
    assert.equal(evidence.rebase.responseChain.restartPreserved, true);
    assert.equal(evidence.rebase.responseChain.finalHistoryComplete, true);
    assert.equal(evidence.usage.continuationTurns.length, 3);
    assert.ok(evidence.usage.observedSavedInputTokens > 0);
    assert.equal(provider.authorizationPresent(), true);
    assert.match(result.artifactSha256, /^[a-f0-9]{64}$/);
    assert.equal(JSON.parse(artifactText).schema, CODEX_REBASE_PROVIDER_SMOKE_EVIDENCE_SCHEMA);
    assert.doesNotMatch(artifactText, /EVICT_ME_|KEEP_ME_|opaque-provider-fixture-/);
    assert.doesNotMatch(artifactText, /provider-smoke-test-key-not-secret/);
    assert.doesNotMatch(artifactText, /previous_response_id|authorization|bearer/i);
  } finally {
    if (previousKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = previousKey;
    await provider.close();
    await rm(outputDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 20 });
  }
});
