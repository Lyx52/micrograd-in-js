import type { ICallable } from "../interfaces/ICallable.ts";
import type {IModule} from "../interfaces/IModule.ts";
import type {Tensor} from "../tensor.ts";
import type {IParameterized, ParameterUpdateCallback} from "../interfaces/IParameterized.ts";
import type {NewTensor} from "../tensor_new.ts";

export class LinearModule implements IModule, ICallable, IParameterized {
    private _layers: ICallable[] = [];

    constructor(...layers: ICallable[]) {
        this._layers = layers;
    }

    public updateParameters(update: ParameterUpdateCallback) {
        for (const layer of (this._layers as unknown[] as IParameterized[])) {
            if (layer.updateParameters) {
                layer.updateParameters(update);
            }
        }
    }

    public zerograd() {
        for (const layer of (this._layers as unknown[] as IParameterized[])) {
            if (layer.zerograd) {
                layer.zerograd();
            }
        }
    }

    execute(inputs: NewTensor): NewTensor {
        return this.forward(inputs);
    }

    forward(input: NewTensor): NewTensor {
        let x = input.clone();
        for (const layer of this._layers) {
            x = layer.execute(x);
        }

        return x;
    }
}