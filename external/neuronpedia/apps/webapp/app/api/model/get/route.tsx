import { getModelByIdWithSourceSets } from '@/lib/db/model';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const { id } = await request.json();
  const model = await getModelByIdWithSourceSets(id, request.user);
  return NextResponse.json(model);
});
