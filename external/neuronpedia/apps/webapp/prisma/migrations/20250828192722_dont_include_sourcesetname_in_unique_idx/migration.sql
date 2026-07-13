/*
  Warnings:

  - A unique constraint covering the columns `[modelId,slug]` on the table `GraphMetadata` will be added. If there are existing duplicate values, this will fail.

*/
-- DropIndex
DROP INDEX "GraphMetadata_modelId_sourceSetName_slug_key";

-- CreateIndex
CREATE UNIQUE INDEX "GraphMetadata_modelId_slug_key" ON "GraphMetadata"("modelId", "slug");
