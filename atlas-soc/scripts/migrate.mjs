import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import pg from "pg";

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL is required to apply the Atlas schema");
}

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === "production" ? { rejectUnauthorized: false } : undefined,
});

try {
  const schema = await readFile(path.join(process.cwd(), "db", "schema.sql"), "utf8");
  await pool.query(schema);
  process.stdout.write("Atlas database schema is current.\n");
} finally {
  await pool.end();
}
