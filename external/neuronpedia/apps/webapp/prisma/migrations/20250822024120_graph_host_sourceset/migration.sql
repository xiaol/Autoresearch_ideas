/*
  Warnings:

  - You are about to drop the column `graphEnabled` on the `Source` table. All the data in the column will be lost.
  - You are about to drop the `GraphHostSourceOnSource` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE "GraphHostSourceOnSource" DROP CONSTRAINT "GraphHostSourceOnSource_graphHostSourceId_fkey";

-- DropForeignKey
ALTER TABLE "GraphHostSourceOnSource" DROP CONSTRAINT "GraphHostSourceOnSource_sourceId_sourceModelId_fkey";

-- AlterTable
ALTER TABLE "Source" DROP COLUMN "graphEnabled";

-- AlterTable
ALTER TABLE "SourceSet" ADD COLUMN     "graphEnabled" BOOLEAN NOT NULL DEFAULT false;

-- DropTable
DROP TABLE "GraphHostSourceOnSource";

-- CreateTable
CREATE TABLE "GraphHostSourceOnSourceSet" (
    "sourceSetName" TEXT NOT NULL,
    "sourceSetModelId" TEXT NOT NULL,
    "graphHostSourceId" TEXT NOT NULL,

    CONSTRAINT "GraphHostSourceOnSourceSet_pkey" PRIMARY KEY ("sourceSetName","sourceSetModelId","graphHostSourceId")
);

-- CreateIndex
CREATE INDEX "GraphHostSourceOnSourceSet_sourceSetModelId_idx" ON "GraphHostSourceOnSourceSet"("sourceSetModelId");

-- CreateIndex
CREATE INDEX "GraphHostSourceOnSourceSet_sourceSetModelId_sourceSetName_idx" ON "GraphHostSourceOnSourceSet"("sourceSetModelId", "sourceSetName");

-- AddForeignKey
ALTER TABLE "GraphHostSourceOnSourceSet" ADD CONSTRAINT "GraphHostSourceOnSourceSet_sourceSetName_sourceSetModelId_fkey" FOREIGN KEY ("sourceSetName", "sourceSetModelId") REFERENCES "SourceSet"("name", "modelId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GraphHostSourceOnSourceSet" ADD CONSTRAINT "GraphHostSourceOnSourceSet_graphHostSourceId_fkey" FOREIGN KEY ("graphHostSourceId") REFERENCES "GraphHostSource"("id") ON DELETE CASCADE ON UPDATE CASCADE;
