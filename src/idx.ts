import {getTotalElements} from "./nn/utils.ts";
import {Tensor} from "./nn/tensor.ts";

const enum IdxTypes {
    UBYTE = 0x08,
    BYTE = 0x09,
    SHORT = 0x0B,
    INT = 0x0C,
    FLOAT = 0x0D,
    DOUBLE = 0x0E,
}

const toIdxType = (value: number): IdxTypes => {
    if (value < 0x08 || value > 0x0E) {
        throw new Error("Invalid value type");
    }

    return value as IdxTypes;
}

const typeToString = (type: IdxTypes): string => {
    switch (type) {
        case IdxTypes.UBYTE: return "UBYTE";
        case IdxTypes.BYTE: return "BYTE";
        case IdxTypes.SHORT: return "SHORT";
        case IdxTypes.INT: return "INT";
        case IdxTypes.FLOAT: return "FLOAT";
        case IdxTypes.DOUBLE: return "DOUBLE";
    }
}

const typeToSize = (type: IdxTypes): number => {
    switch (type) {
        case IdxTypes.UBYTE: return 1;
        case IdxTypes.BYTE: return 1;
        case IdxTypes.SHORT: return 2;
        case IdxTypes.INT: return 4;
        case IdxTypes.FLOAT: return 4;
        case IdxTypes.DOUBLE: return 8;
    }
}

class BufferReader {
    private position: number = 0;
    private buffer: Uint8Array<ArrayBuffer> = null;
    private view: DataView = null;
    constructor(buffer: Uint8Array<ArrayBuffer>) {
        this.buffer = buffer;
        this.view = new DataView(buffer.buffer, 0);
        this.position = 0;
    }

    public readByte() {
        let result = this.view.getInt8(this.position);
        this.position++;

        return result;
    }

    public readUByte() {
        return this.buffer[this.position++];
    }

    public readShort() {
        let result = this.view.getInt16(this.position);
        this.position += 2;

        return result;
    }

    public readInt() {
        let result = this.view.getInt32(this.position);
        this.position += 4;

        return result;
    }

    public readFloat() {
        let result = this.view.getFloat32(this.position);
        this.position += 4;

        return result;
    }

    public readDouble() {
        let result = this.view.getFloat64(this.position);
        this.position += 8;

        return result;
    }

    public readInts(n: number) {
        const result = [];
        for (let i = 0; i < n; i++) {
            result.push(this.readInt());
        }

        return result;
    }
}
export const readIdx = async (file: () => Promise<Uint8Array<ArrayBuffer>>, readDims: number[] = []) => {
    const buffer = await file();

    const reader = new BufferReader(buffer);
    if (reader.readUByte() !== 0 || reader.readUByte() !== 0) {
        throw new Error('Expected magic 0x00');
    }

    const dataType = toIdxType(reader.readUByte());
    const dimsCount = reader.readUByte();
    let dims = reader.readInts(dimsCount);
    for (let i = 0; i < readDims.length; i++) {
        if (readDims[i] < 0) continue;
        dims[i] = Math.min(dims[i], readDims[i]);
    }
    const total = getTotalElements(...dims);
    const data = [];
    for (let i = 0; i < total; i++) {
        switch (dataType) {
            case IdxTypes.UBYTE:
                data.push(reader.readUByte());
                break;
            case IdxTypes.BYTE:
                data.push(reader.readByte());
                break;
            case IdxTypes.SHORT:
                data.push(reader.readShort());
                break;
            case IdxTypes.INT:
                data.push(reader.readInt());
                break;
            case IdxTypes.FLOAT:
                data.push(reader.readFloat());
                break;
            case IdxTypes.DOUBLE:
                data.push(reader.readDouble());
                break;
        }
    }

    return [dims as number[], data as number[]];
}