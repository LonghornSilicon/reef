import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { GetObjectCommand, PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { isLocalDevelopment } from "@/lib/config";

const localRoot = process.env.ATLAS_LOCAL_UPLOAD_DIR ?? "/tmp/atlas-uploads";

function s3Configured() {
  return Boolean(process.env.S3_BUCKET && process.env.S3_REGION);
}

function s3() {
  return new S3Client({
    region: process.env.S3_REGION,
    endpoint: process.env.S3_ENDPOINT || undefined,
    forcePathStyle: Boolean(process.env.S3_ENDPOINT),
    credentials:
      process.env.S3_ACCESS_KEY_ID && process.env.S3_SECRET_ACCESS_KEY
        ? {
            accessKeyId: process.env.S3_ACCESS_KEY_ID,
            secretAccessKey: process.env.S3_SECRET_ACCESS_KEY,
          }
        : undefined,
  });
}

export async function putObject(key: string, bytes: Uint8Array, contentType: string) {
  if (s3Configured()) {
    await s3().send(
      new PutObjectCommand({
        Bucket: process.env.S3_BUCKET,
        Key: key,
        Body: bytes,
        ContentType: contentType,
        ServerSideEncryption: "AES256",
      }),
    );
    return;
  }
  // The local-disk fallback is for a developer's machine only. On a host it
  // would accept an upload, write it to a container filesystem that is
  // discarded on the next deploy, and report success.
  if (!isLocalDevelopment()) throw new Error("S3-compatible document storage is not configured");
  const target = path.join(/* turbopackIgnore: true */ localRoot, key);
  if (!target.startsWith(`${localRoot}${path.sep}`)) throw new Error("Invalid storage key");
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, bytes);
}

export async function getObject(key: string): Promise<Uint8Array> {
  if (s3Configured()) {
    const result = await s3().send(new GetObjectCommand({ Bucket: process.env.S3_BUCKET, Key: key }));
    if (!result.Body) throw new Error("Stored document has no body");
    return result.Body.transformToByteArray();
  }
  const target = path.join(/* turbopackIgnore: true */ localRoot, key);
  if (!target.startsWith(`${localRoot}${path.sep}`)) throw new Error("Invalid storage key");
  return readFile(target);
}
