import kNear from "./knear";

const k = 3;

const machine = new kNear(k);

machine.learn([6, 5, 9], 'cat');
machine.learn([12, 20, 19], 'dog');
machine.learn([9.2, 8.1, 2],'cat');
machine.learn([17, 15.5, 5], 'dog');
machine.learn([9.1, 9, 1.95], 'cat');
machine.learn([20, 20, 6.2], 'dog');
machine.learn([9.0, 10, 2.1], 'cat');
machine.learn([16.7, 16, 3.3], 'dog');

let prediction = machine.classify([12, 18, 17]);

console.log(`I think this is a ${prediction}!`);