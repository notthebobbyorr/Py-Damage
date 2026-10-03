export default function(component) {
    const {parentElement, data} = component;
    const root = parentElement.querySelector('.table-scroll');
    const toolbar = parentElement.querySelector('.table-toolbar');
    const button = toolbar.querySelector('.fullscreen-toggle');
    const dialog = parentElement.querySelector('.table-fullscreen');
    const home = dialog.parentNode;
    const restore = () => {
        home.insertBefore(toolbar, dialog);
        home.insertBefore(root, dialog);
        button.textContent = 'Fullscreen';
        button.setAttribute('aria-expanded', 'false');
        button.focus();
    };
    button.setAttribute('aria-expanded', String(dialog.open));
    button.onclick = () => {
        if (dialog.open) {
            dialog.close();
        } else {
            dialog.append(toolbar, root);
            button.textContent = 'Exit fullscreen';
            button.setAttribute('aria-expanded', 'true');
            dialog.showModal();
            button.focus();
        }
    };
    dialog.onclose = restore;
    root.innerHTML = data.html;
    const table = root.querySelector('table');
    const body = table.tBodies[0];
    const rows = [...body.rows];
    const originalCells = new Map(rows.map(row => [row, [...row.cells]]));
    rows.forEach((row, index) => {
        row.classList.add('player');
        row.tabIndex = 0;
        row.setAttribute('aria-expanded', 'false');
        row.setAttribute('aria-label', `${row.cells[0].textContent}: toggle official season stats`);
        const detail = document.createElement('tr');
        detail.className = 'detail';
        detail.hidden = true;
        const cell = detail.insertCell();
        cell.colSpan = row.cells.length;
        const values = data.details[index];
        if (values) {
            const inner = document.createElement('table');
            inner.setAttribute('aria-label', 'Official season totals');
            const headings = inner.insertRow();
            const totals = inner.insertRow();
            data.columns.forEach((name, i) => {
                const th = document.createElement('th');
                th.textContent = name;
                headings.appendChild(th);
                totals.insertCell().textContent = values[i];
            });
            cell.appendChild(inner);
        } else {
            cell.textContent = 'Official regular-season totals for this level are not available for this row.';
        }
        row.after(detail);
        const toggle = () => {
            detail.hidden = !detail.hidden;
            row.setAttribute('aria-expanded', String(!detail.hidden));
        };
        row.onclick = toggle;
        row.onkeydown = event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                toggle();
            }
        };
        row.detailRow = detail;
    });
    [...table.tHead.rows[0].cells].forEach((header, column) => {
        let ascending = false;
        header.onclick = () => {
            ascending = !ascending;
            rows.sort((a, b) => {
                const left = originalCells.get(a)[column].textContent;
                const right = originalCells.get(b)[column].textContent;
                const comparison = left.trim() && right.trim() && Number.isFinite(Number(left)) && Number.isFinite(Number(right))
                    ? Number(left) - Number(right) : left.localeCompare(right);
                return ascending ? comparison : -comparison;
            });
            rows.forEach(row => body.append(row, row.detailRow));
        };
    });
    const headers = [...table.tHead.rows[0].cells];
    let order = headers.map((_, i) => i);
    const hidden = new Set();
    const options = toolbar.querySelector('.column-options');
    const applyColumns = () => {
        const headerRow = table.tHead.rows[0];
        order.forEach(i => {
            headers[i].hidden = hidden.has(i);
            headerRow.append(headers[i]);
            rows.forEach(row => {
                const cell = originalCells.get(row)[i];
                cell.hidden = hidden.has(i);
                row.append(cell);
            });
        });
        rows.forEach(row => row.detailRow.cells[0].colSpan = order.length - hidden.size);
    };
    const renderOptions = () => {
        options.replaceChildren();
        const reset = document.createElement('button');
        reset.type = 'button';
        reset.textContent = 'Reset columns';
        reset.onclick = () => {
            order = headers.map((_, i) => i);
            hidden.clear();
            applyColumns();
            renderOptions();
        };
        options.append(reset);
        order.forEach((i, position) => {
            const item = document.createElement('div');
            item.className = 'column-option';
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = !hidden.has(i);
            checkbox.disabled = checkbox.checked && hidden.size === order.length - 1;
            checkbox.onchange = () => {
                if (checkbox.checked) hidden.delete(i); else hidden.add(i);
                applyColumns();
                renderOptions();
            };
            label.append(checkbox, document.createTextNode(headers[i].textContent));
            item.append(label);
            [-1, 1].forEach(direction => {
                const move = document.createElement('button');
                move.type = 'button';
                move.textContent = direction === -1 ? '↑' : '↓';
                move.setAttribute('aria-label', `Move ${headers[i].textContent} ${direction === -1 ? 'left' : 'right'}`);
                move.disabled = position + direction < 0 || position + direction >= order.length;
                move.onclick = () => {
                    [order[position], order[position + direction]] = [order[position + direction], order[position]];
                    applyColumns();
                    renderOptions();
                };
                item.append(move);
            });
            options.append(item);
        });
    };
    renderOptions();

}
