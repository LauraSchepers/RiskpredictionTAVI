// Ensure the script runs after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function () {
    document.querySelector('button').addEventListener('click', calculateMortality);
});

function calculateMortality() {
    const age = parseFloat(document.getElementById('age').value);
    const lengthCM = parseFloat(document.getElementById('lengthCM').value);
    const weight = parseFloat(document.getElementById('weight').value);
    const BL_AF = document.getElementById('BL_AF').checked ? 1 : 0;
    const HistoryAorticValveIntervention = document.getElementById('HistoryAorticValveIntervention').checked ? 1 : 0;
    const NYHA = parseInt(document.getElementById('NYHA').value, 10);
    const eGFR = parseFloat(document.getElementById('eGFR').value);
    const LVEF = parseFloat(document.getElementById('LVEF').value);
    const AorticValveArea = parseFloat(document.getElementById('AorticValveArea').value);

    const data = {
        age,
        lengthCM,
        weight,
        BL_AF,
        HistoryAorticValveIntervention,
        NYHA,
        eGFR,
        LVEF,
        AorticValveArea
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById('mortality-chance').innerText = `${result.mortality_chance} (${(result.probability * 100).toFixed(2)}%)`;
    })
    .catch(error => console.error('Error:', error));
}
