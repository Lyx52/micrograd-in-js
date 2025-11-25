import type {ICallable} from "../interfaces/ICallable.ts";
import {Layer} from "./layer.ts";
import type {IParameterized, ParameterUpdateCallback} from "../interfaces/IParameterized.ts";
import {NewTensor} from "../tensor_new.ts";

export class MLP implements ICallable, IParameterized {
    private layers: Layer[] = [];

    constructor(inputs: number, outputs: number[]) {
        const sizes = [inputs, ...outputs];
        for (let i = 0; i < outputs.length; i++) {
            this.layers.push(new Layer(sizes[i], sizes[i + 1]))
        }
    }

    public updateParameters(update: ParameterUpdateCallback) {
        for (const layer of this.layers) {
            layer.updateParameters(update);
        }
    }

    public zerograd() {
        for (const layer of this.layers) {
            layer.zerograd();
        }
    }

    public execute(inputs: NewTensor): NewTensor {
        let layerInput = inputs;
        for (const layer of this.layers) {
            layerInput = layer.execute(layerInput);
        }

        return layerInput;
    }
}