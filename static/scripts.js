document.querySelector('form')?.addEventListener('submit', function(e) {
    const inputs = this.querySelectorAll('input[required]');
    let valid = true;
    inputs.forEach(input => {
        if (!input.value) {
            valid = false;
            alert(`${input.id} is required!`);
        }
    });
    if (!valid) e.preventDefault();
});