import { Pool, type PoolClient, type QueryResultRow } from "pg";

declare global {
  var __atlasPool: Pool | undefined;
}

export function hasDatabase() {
  return Boolean(process.env.DATABASE_URL);
}

function getPool() {
  if (!process.env.DATABASE_URL) return null;
  if (!globalThis.__atlasPool) {
    globalThis.__atlasPool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 10,
      ssl: process.env.NODE_ENV === "production" ? { rejectUnauthorized: false } : undefined,
    });
  }
  return globalThis.__atlasPool;
}

export async function query<T extends QueryResultRow>(text: string, values: unknown[] = []) {
  const pool = getPool();
  if (!pool) throw new Error("DATABASE_URL is not configured");
  return pool.query<T>(text, values);
}

export async function transaction<T>(work: (client: PoolClient) => Promise<T>) {
  const pool = getPool();
  if (!pool) throw new Error("DATABASE_URL is not configured");
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await work(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
