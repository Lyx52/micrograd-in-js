import {MLP} from "../../nn/mlp/mlp.ts";
import {crossEntropyLoss} from "../../nn/utils.ts";
import {NewTensor} from "../../nn/tensor_new.ts";

export const runSimpleMLP = () => {
    const mlp = new MLP(3, [4, 4, 1]);
    const xs = [
        NewTensor.from([2.0, 3.0, -1.0]),
        NewTensor.from([3.0, -1.0, 0.5]),
        NewTensor.from([0.5, 1.0, 1.0]),
        NewTensor.from([1.0, 1.0, -1.0]),
    ];
    const ys = [
        NewTensor.from([1.0]),
        NewTensor.from([-1.0]),
        NewTensor.from([-1.0]),
        NewTensor.from([1.0]),
    ];

    // Train
    const learningRate = 0.001;
    const steps = 10000;
    for (let i = 0; i <= steps; i++) {
        mlp.updateParameters((tensor: NewTensor) => {
            for (let j = 0; j < tensor.backing.length; j++) {
                tensor.backing[j] += -learningRate * tensor.gradients[j];
            }
        });

        const loss = crossEntropyLoss(mlp, ys, xs);
        mlp.zerograd();
        loss.backward();
        console.log(`[Step ${i}/${steps}] loss: ${loss.scalar()}`)
    }
}