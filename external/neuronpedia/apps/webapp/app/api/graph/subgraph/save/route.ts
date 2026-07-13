import { SaveSubgraphRequestSchema } from '@/app/[modelId]/graph/graph-types';
import { ANTHROPIC_MODELS } from '@/app/[modelId]/graph/utils';
import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/graph/subgraph/save:
 *   post:
 *     summary: Subgraph Save (or Overwrite)
 *     description: Creates a new subgraph or overwrites an existing one if overwriteId is provided
 *     tags:
 *       - Attribution Graphs
 *     security:
 *       - apiKey: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - modelId
 *               - slug
 *               - pinnedIds
 *               - supernodes
 *               - clerps
 *             properties:
 *               modelId:
 *                 type: string
 *                 example: "gemma-2-2b"
 *               slug:
 *                 type: string
 *                 example: "my doggo graph"
 *               displayName:
 *                 type: string
 *                 example: "my cool subgraph"
 *               pinnedIds:
 *                 type: array
 *                 items:
 *                   type: string
 *                 example: ["2_15681_2", "E_2_0", "4_14735_2", "E_5929_2", "19_9180_3", "27_6784_5"]
 *               supernodes:
 *                 type: array
 *                 items:
 *                   type: array
 *                   items:
 *                     type: string
 *                 example: [["supernode", "4_14735_2", "19_9180_3"]]
 *               clerps:
 *                 type: array
 *                 items:
 *                   type: object
 *                 example: []
 *               pruningThreshold:
 *                 type: number
 *                 nullable: true
 *                 example: 0.8
 *               densityThreshold:
 *                 type: number
 *                 nullable: true
 *                 example: 0.99
 *               overwriteId:
 *                 type: string
 *                 description: ID of existing subgraph to overwrite (optional)
 *     responses:
 *       200:
 *         description: Subgraph saved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 subgraphId:
 *                   type: string
 *       403:
 *         description: Not authorized to overwrite this subgraph
 *       500:
 *         description: Failed to save subgraph
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const body = await request.json();
    const validatedData = SaveSubgraphRequestSchema.parse(body);
    const {
      modelId,
      slug,
      displayName,
      pinnedIds,
      supernodes,
      clerps,
      pruningThreshold,
      densityThreshold,
      overwriteId,
    } = validatedData;

    // if overwriteId, check that the logged in user equals the owner of the subgraph
    if (overwriteId) {
      const subgraph = await prisma.graphMetadataSubgraph.findUnique({
        where: { id: overwriteId },
      });
      if (subgraph?.userId !== request.user.id) {
        return NextResponse.json({ error: 'You are not authorized to overwrite this subgraph' }, { status: 403 });
      }
      // user is the owner, update the subgraph
      await prisma.graphMetadataSubgraph.update({
        where: { id: overwriteId },
        data: {
          pinnedIds,
          supernodes,
          clerps,
          pruningThreshold,
          densityThreshold,
        },
      });
      return NextResponse.json({ success: true, subgraphId: overwriteId });
    }

    // if this is an anthropic model, we don't actually have the graph in our database since we load it from anthropic's site.
    // so check if the graph metadata record exists. if not, create a stub for it.
    if (ANTHROPIC_MODELS.has(modelId)) {
      await prisma.graphMetadata.upsert({
        where: {
          modelId_slug: {
            modelId,
            slug,
          },
        },
        create: {
          modelId,
          slug,
          prompt: '',
          titlePrefix: '',
          url: '',
        },
        update: {
          prompt: '',
          titlePrefix: '',
          url: '',
        },
      });
    }

    // save the subgraph
    const subgraph = await prisma.graphMetadataSubgraph.create({
      data: {
        displayName,
        graphMetadata: {
          connect: {
            modelId_slug: {
              modelId,
              slug,
            },
          },
        },
        pinnedIds,
        supernodes,
        clerps,
        pruningThreshold,
        densityThreshold,
        user: {
          connect: {
            id: request.user.id,
          },
        },
      },
    });
    return NextResponse.json({ success: true, subgraphId: subgraph.id });
  } catch (error) {
    console.error('Error saving subgraph:', error);
    return NextResponse.json({ error: 'Failed to save subgraph' }, { status: 500 });
  }
});
