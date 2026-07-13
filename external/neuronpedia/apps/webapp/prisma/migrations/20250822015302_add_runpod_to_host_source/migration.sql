-- AlterTable
ALTER TABLE "GraphHostSource" ADD COLUMN     "runpodServerlessUrl" TEXT,
ALTER COLUMN "hostUrl" DROP NOT NULL;
