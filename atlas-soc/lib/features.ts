/**
 * Feature state that both the server and the browser bundle need to read.
 *
 * Separate from lib/config.ts because that module imports node:crypto, which
 * cannot be bundled into a client component. Nothing here may import Node
 * built-ins or read process.env.
 */

/**
 * Atlas ships with no AI integration.
 *
 * Deliberately a constant rather than an environment flag: setting
 * OPENAI_API_KEY on a host must not be enough to start calling a paid API.
 * Flipping this one line is the only way to turn the integration on, and two
 * things have to be true first.
 *
 * 1. lib/rate-limit.ts keeps its buckets in an in-process Map, which bounds
 *    spend only while there is exactly one long-lived process. On a
 *    serverless host every instance gets its own Map and nothing binds.
 * 2. The key belongs in the host's secret store, never in this repository
 *    or in a committed env file.
 *
 * Every surface reads this through aiIntegrationEnabled() and says plainly
 * that the integration is absent, rather than offering a control that fails.
 */
const AI_INTEGRATION = false;

export function aiIntegrationEnabled() {
  return AI_INTEGRATION;
}
