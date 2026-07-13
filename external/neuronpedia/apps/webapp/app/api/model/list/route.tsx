import { getGlobalModels } from '@/lib/db/model';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const models = await getGlobalModels(request.user);
  return NextResponse.json(models);
});
