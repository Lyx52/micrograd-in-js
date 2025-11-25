import type {NewTensor} from "../tensor_new.ts";

export interface ICallable {
    execute(inputs: NewTensor): NewTensor;
}