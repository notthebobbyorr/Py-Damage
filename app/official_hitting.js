export default function(component) {
    const {parentElement, data} = component;
    const root = parentElement.querySelector('.table-scroll');
    root.innerHTML = data.html;
    const table = root.querySelector('table');
    const body = table.tBodies[0];
    const rows = [...body.rows];
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
                const left = a.cells[column].textContent;
                const right = b.cells[column].textContent;
                const comparison = left.trim() && right.trim() && Number.isFinite(Number(left)) && Number.isFinite(Number(right))
                    ? Number(left) - Number(right) : left.localeCompare(right);
                return ascending ? comparison : -comparison;
            });
            rows.forEach(row => body.append(row, row.detailRow));
        };
    });
}
