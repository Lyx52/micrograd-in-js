import type {NewTensor} from "../tensor_new.ts";

export type ParameterUpdateCallback = (tensor: NewTensor) => void;

export interface IParameterized {
    updateParameters(update: ParameterUpdateCallback);
    zerograd();
}