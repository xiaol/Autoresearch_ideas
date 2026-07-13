/*
  Warnings:

  - A unique constraint covering the columns `[modelId,sourceSetName,slug]` on the table `GraphMetadata` will be added. If there are existing duplicate values, this will fail.

*/
-- DropIndex
DROP INDEX "GraphMetadata_modelId_slug_key";

-- AlterTable
ALTER TABLE "GraphMetadata" ADD COLUMN     "sourceSetName" TEXT;

-- CreateIndex
CREATE INDEX "GraphMetadata_modelId_sourceSetName_idx" ON "GraphMetadata"("modelId", "sourceSetName");

-- CreateIndex
CREATE UNIQUE INDEX "GraphMetadata_modelId_sourceSetName_slug_key" ON "GraphMetadata"("modelId", "sourceSetName", "slug");

-- AddForeignKey
ALTER TABLE "GraphMetadata" ADD CONSTRAINT "GraphMetadata_modelId_sourceSetName_fkey" FOREIGN KEY ("modelId", "sourceSetName") REFERENCES "SourceSet"("modelId", "name") ON DELETE CASCADE ON UPDATE CASCADE;
