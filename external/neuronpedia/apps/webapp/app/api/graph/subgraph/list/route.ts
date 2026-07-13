import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/graph/subgraph/list:
 *   post:
 *     summary: Subgraphs List
 *     description: Lists all subgraphs owned by the authenticated user for a specific graph
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
 *             properties:
 *               modelId:
 *                 type: string
 *                 example: "gemma-2-2b"
 *               slug:
 *                 type: string
 *                 example: "thedogsitsonthe-1756413232594"
 *     responses:
 *       200:
 *         description: Subgraphs retrieved successfully
 *       500:
 *         description: Failed to retrieve subgraphs
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const { modelId, slug } = await request.json();

    // list subgraphs that are owned by the user for this graph
    const subgraphs = await prisma.graphMetadataSubgraph.findMany({
      where: {
        userId: request.user.id,
        graphMetadata: {
          modelId,
          slug,
        },
      },
      include: {
        user: {
          select: {
            name: true,
            id: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({ success: true, subgraphs });
  } catch (error) {
    console.error('Error saving subgraph:', error);
    return NextResponse.json({ error: 'Failed to save subgraph' }, { status: 500 });
  }
});
