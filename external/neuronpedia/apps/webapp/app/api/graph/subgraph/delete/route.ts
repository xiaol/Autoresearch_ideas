import { DeleteSubgraphRequestSchema } from '@/app/[modelId]/graph/graph-types';
import { prisma } from '@/lib/db';
import { RequestAuthedUser, withAuthedUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * @swagger
 * /api/graph/subgraph/delete:
 *   post:
 *     summary: Subgraph Delete
 *     description: Deletes an existing subgraph owned by the authenticated user
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
 *               - subgraphId
 *             properties:
 *               subgraphId:
 *                 type: string
 *                 description: ID of the subgraph to delete
 *                 example: "clx1234567890abcdef"
 *     responses:
 *       200:
 *         description: Subgraph deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: "Subgraph deleted successfully"
 *       403:
 *         description: Not authorized to delete this subgraph
 *       404:
 *         description: Subgraph not found
 *       500:
 *         description: Failed to delete subgraph
 */

export const POST = withAuthedUser(async (request: RequestAuthedUser) => {
  try {
    const body = await request.json();
    const validatedData = DeleteSubgraphRequestSchema.parse(body);
    const { subgraphId } = validatedData;

    // Find the subgraph to check ownership
    const subgraph = await prisma.graphMetadataSubgraph.findUnique({
      where: { id: subgraphId },
    });

    if (!subgraph) {
      return NextResponse.json({ error: 'Subgraph not found' }, { status: 404 });
    }

    // Check that the logged in user equals the owner of the subgraph
    if (subgraph.userId !== request.user.id) {
      return NextResponse.json({ error: 'You are not authorized to delete this subgraph' }, { status: 403 });
    }

    // Delete the subgraph
    await prisma.graphMetadataSubgraph.delete({
      where: { id: subgraphId },
    });

    return NextResponse.json({ success: true, message: 'Subgraph deleted successfully' });
  } catch (error) {
    console.error('Error deleting subgraph:', error);
    return NextResponse.json({ error: 'Failed to delete subgraph' }, { status: 500 });
  }
});
