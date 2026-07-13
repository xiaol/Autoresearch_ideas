import { prisma } from '@/lib/db';
import { ModelWithRelations } from '@/prisma/generated/zod';
import AvailableResourcesTable from './table';

export default async function AvailableResourcesPage() {
  const models = await prisma.model.findMany({
    where: {
      visibility: 'PUBLIC',
    },
    include: {
      sourceSets: {
        where: { visibility: 'PUBLIC', hasDashboards: true },
        include: {
          sources: {
            where: { visibility: 'PUBLIC' },
            orderBy: { id: 'asc' },
            select: { id: true, inferenceEnabled: true },
          },
        },
        orderBy: { name: 'asc' },
      },
    },
    orderBy: {
      id: 'asc',
    },
  });

  return <AvailableResourcesTable models={models as ModelWithRelations[]} />;
}
