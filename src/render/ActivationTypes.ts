import type {IOption} from "../nn/utils.ts";

export enum ActivationType {
    Tanh = 'tanh',
    ReLu = 'relu',
    None = 'none',
}

export function toActivationName(type: ActivationType) {
    switch (type) {
        case ActivationType.ReLu:
            return 'ReLu';
        case ActivationType.Tanh:
            return 'Tanh';
        case ActivationType.None:
            return 'None';
        default:
            return '';
    }
}

export const getActivationOptions = (): IOption<ActivationType>[] => {
    const options = [ActivationType.ReLu, ActivationType.Tanh, ActivationType.None];

    return options.map(v => ({ value: v, text: toActivationName(v) }));
}