/*
  Warnings:

  - A unique constraint covering the columns `[defaultGraphSourceSetName,id]` on the table `Model` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterTable
ALTER TABLE "Model" ADD COLUMN     "defaultGraphSourceSetName" TEXT;

-- AlterTable
ALTER TABLE "SourceSet" ADD COLUMN     "defaultGraphModelId" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX "Model_defaultGraphSourceSetName_id_key" ON "Model"("defaultGraphSourceSetName", "id");

-- AddForeignKey
ALTER TABLE "Model" ADD CONSTRAINT "Model_defaultGraphSourceSetName_id_fkey" FOREIGN KEY ("defaultGraphSourceSetName", "id") REFERENCES "SourceSet"("name", "modelId") ON DELETE CASCADE ON UPDATE CASCADE;
