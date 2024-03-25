import kNear from "./knear.js";

const k = 3;

const machine = new kNear(k);
let filename;


const filenames = ['../Victory.json', '../L.json', '../Metal.json'];
const labels = ['Victory', 'L', 'Metal'];


async function fetchJSONFile(filename) {
    try {
        const response = await fetch(filename);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${filename}: ${response.status} ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${filename}: ${error}`);
        return [];
    }
}

async function trainMachine() {
    for (let i = 0; i < filenames.length; i++) {

        filename = filenames[i];
        const label = labels[i];
        const data = await fetchJSONFile(filename);
        data.forEach(entry => {
            // Extract pose data from each entry
            const pose = entry.pose;
            // Train the machine with the pose data and label
            machine.learn(pose, label);
        });
    }
}

// Call the trainMachine function to start training
trainMachine().then(() => {
    // Example classification
    let prediction = machine.classify([12, 18, 17]);
    console.log(`I think this is a ${prediction}!`);
}).catch(error => {
    console.error('Failed to train machine:', error);
});

// machine.learn([6, 5, 9], 'cat');
// machine.learn([12, 20, 19], 'dog');
// machine.learn([9.2, 8.1, 2],'cat');
// machine.learn([17, 15.5, 5], 'dog');
// machine.learn([9.1, 9, 1.95], 'cat');
// machine.learn([20, 20, 6.2], 'dog');
// machine.learn([9.0, 10, 2.1], 'cat');
// machine.learn([16.7, 16, 3.3], 'dog');
//
// let prediction = machine.classify([12, 18, 17]);
