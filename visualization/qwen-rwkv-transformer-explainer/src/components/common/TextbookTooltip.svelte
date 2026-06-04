<script lang="ts">
	import {
		isTextbookOpen,
		textbookCurrentPage,
		textbookCurrentPageId,
		textbookPreviousPage,
		textbookPreviousPageId,
		selectedTraceModel,
		userId
	} from '~/store';
	import { getTextPagesForModel, mapTextPageIdForModel } from '~/utils/textbookPages';

	export let id: string;

	function openTextbook(e) {
		e.stopPropagation();
		e.preventDefault();

		const activeTextPages = getTextPagesForModel($selectedTraceModel);
		const mappedId = mapTextPageIdForModel(id, $selectedTraceModel);
		const pageIndex = activeTextPages.findIndex((page) => page.id === mappedId);
		if (pageIndex !== -1) {
			textbookPreviousPageId.set($textbookCurrentPageId);
			textbookPreviousPage.set($textbookCurrentPage);
			isTextbookOpen.set(true);
			textbookCurrentPage.set(pageIndex);
			textbookCurrentPageId.set(mappedId);
		}

		window.dataLayer?.push({
			event: `open-textbook`,
			page_id: pageIndex === -1 ? id : mappedId,
			open_via: 'tooltip',
			user_id: $userId
		});
	}
</script>

<div
	{id}
	class="textbook-tooltip"
	data-click={`textbook-tooltip`}
	on:click={openTextbook}
	role="button"
	tabindex="0"
	on:keydown={(e) => {
		if (e.key === 'Enter' || e.key === ' ') {
			openTextbook(e);
		}
	}}
>
	<slot></slot>
</div>

<style lang="scss">
	.textbook-tooltip {
		cursor: help;
	}
</style>
