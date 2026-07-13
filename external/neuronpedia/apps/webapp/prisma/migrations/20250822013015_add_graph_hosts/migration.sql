-- AlterTable
ALTER TABLE "Source" ADD COLUMN     "graphEnabled" BOOLEAN NOT NULL DEFAULT false;

-- CreateTable
CREATE TABLE "GraphHostSource" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "hostUrl" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "GraphHostSource_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "GraphHostSourceOnSource" (
    "sourceId" TEXT NOT NULL,
    "sourceModelId" TEXT NOT NULL,
    "graphHostSourceId" TEXT NOT NULL,

    CONSTRAINT "GraphHostSourceOnSource_pkey" PRIMARY KEY ("sourceId","sourceModelId","graphHostSourceId")
);

-- CreateIndex
CREATE INDEX "GraphHostSource_modelId_idx" ON "GraphHostSource"("modelId");

-- CreateIndex
CREATE INDEX "GraphHostSourceOnSource_sourceModelId_idx" ON "GraphHostSourceOnSource"("sourceModelId");

-- CreateIndex
CREATE INDEX "GraphHostSourceOnSource_sourceModelId_sourceId_idx" ON "GraphHostSourceOnSource"("sourceModelId", "sourceId");

-- AddForeignKey
ALTER TABLE "GraphHostSource" ADD CONSTRAINT "GraphHostSource_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GraphHostSourceOnSource" ADD CONSTRAINT "GraphHostSourceOnSource_sourceId_sourceModelId_fkey" FOREIGN KEY ("sourceId", "sourceModelId") REFERENCES "Source"("id", "modelId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GraphHostSourceOnSource" ADD CONSTRAINT "GraphHostSourceOnSource_graphHostSourceId_fkey" FOREIGN KEY ("graphHostSourceId") REFERENCES "GraphHostSource"("id") ON DELETE CASCADE ON UPDATE CASCADE;
