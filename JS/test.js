const nn = ml5.neuralNetwork({ task: 'classification', debug: true,
    layers: [
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        }, {
            type: 'dense',
            units: 32,
            activation: 'relu',
        }]});

const labels = ['Victory', 'L', 'Metal'];
let testData = [];

getFromLocalStorage();

async function getFromLocalStorage () {
    for (let i = 0; i < labels.length; i++) {
        testData = [...testData, ...JSON.parse(localStorage.getItem(`testData ${i}`))];
    }
}

async function testMachine() {

    const modelDetails = {
        model: '../model/model.json',
        metadata: '../model/model_meta.json',
        weights: '../model/model.weights.bin'
    }

    await new Promise((resolve, reject) => {
        nn.load(modelDetails, (err) => {
            if (err) {
                reject(err);
            } else {
                console.log("Model Loaded!");
                resolve();
            }
        });
    });

    await tester(testData, labels);
    console.log(testData);
    await finishedTesting();
}

async function tester (data) {
    data.forEach(entry => {
        const pose = entry.pose;
        console.log(entry.label);
        nn.addData(pose, { label: entry.label });
    });
}

async function finishedTesting() {

    //const confusionMatrix = Array(labels.length).fill(0).map(() => Array(labels.length).fill(0));
    let correct = 0;
    let total = 0;
    for (const entry of testData) {
        const pose = entry.pose;
        const label = entry.label;

        const prediction = await nn.classify(pose);

        if (prediction[0].label === label) {
            correct++;
        }
        total++;

        let percentage = correct / total * 100;
        console.log('prediction = ' + prediction + ' ' + percentage);

        //const confusionMatrixTable = generateConfusionMatrixTable(confusionMatrix, labels);

        // Add table to document
        //document.body.innerHTML += confusionMatrixTable;

    }}

    console.log("Finished training!");

// function generateConfusionMatrixTable(confusionMatrix, labels) {
//     let html = '<table>';
//
//     // Add header row
//     html += '<tr><th></th>';
//     for (let i = 0; i < labels.length; i++) {
//         html += `<th>${labels[i]}</th>`;
//     }
//     html += '</tr>';
//
//     // Add data rows
//     for (let i = 0; i < confusionMatrix.length; i++) {
//         html += `<tr><th>${labels[i]}</th>`;
//         for (let j = 0; j < confusionMatrix[i].length; j++) {
//             html += `<td>${confusionMatrix[i][j]}</td>`;
//         }
//         html += '</tr>';
//     }
//
//     html += '</table>';
//
//     return html;
// }

testMachine();
