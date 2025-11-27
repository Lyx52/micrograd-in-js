<template>
  <BRow cols="1">
    <BCol class="mb-4 border-1">
      <BCard
          class="list-group-item cursor-move border-1 shadow-sm user-select-none"
      >
        <div class="d-flex justify-content-between gap-2">
          <BButton variant="success" @click="runNetwork">
            Run network
          </BButton>
        </div>
      </BCard>
    </BCol>
    <BCol class="mb-4 border-1">
      <BCard
          class="list-group-item cursor-move border-1 shadow-sm user-select-none"
      >
        <div class="d-flex justify-content-between gap-2">
          <select class="form-select" v-model="layer.type">
            <option v-for="item in layerOptions" :key="item.value" :value="item.value">
              {{ item.text }}
            </option>
          </select>
          <BFormInput
              v-model="layer.inputs"
              type="number"
              placeholder="Inputs"
          />
          <BFormInput
              v-model="layer.outputs"
              type="number"
              placeholder="Outputs"
          />
          <BButton variant="success" @click="addLayer">
            +
          </BButton>
        </div>
      </BCard>
    </BCol>
    <BCol>
      <NetworkRenderer :layers="this.list" />
    </BCol>
    <BCol>
      <div class="flex justify-center">
        <draggable class="dragArea list-group w-full gap-3" v-model="list" :sort="true">
          <BCard
              class="list-group-item cursor-move border-1 shadow-sm user-select-none"
              v-for="(element, index) in list"
              :key="element.id"
          >
            <div class="d-flex justify-content-between gap-2">
              <select class="form-select" v-model="element.type">
                <option v-for="item in layerOptions" :key="item.value" :value="item.value">
                  {{ item.text }}
                </option>
              </select>
              <BFormInput
                  v-model="element.inputs"
                  type="number"
                  placeholder="Inputs"
              />
              <BFormInput
                  v-model="element.outputs"
                  type="number"
                  placeholder="Outputs"
              />
              <BButton variant="danger" @click="() => removeLayer(element.id)">
                X
              </BButton>
            </div>
          </BCard>
        </draggable>
      </div>
    </BCol>
  </BRow>
</template>

<script lang="ts">
import {defineComponent} from 'vue'
import {VueDraggableNext} from "vue-draggable-next";
import {BButton, BCard, BCol, BFormInput, BListGroup, BListGroupItem, BRow} from "bootstrap-vue-next";
import {NetworkLayer} from "../render/NetworkLayer.ts";
import {getLayerOptions, LayerType} from "../render/LayerTypes.ts";
import {LinearModule} from "../nn/module/linear_module.ts";
import {Flatten} from "../nn/layers/flatten.ts";
import {Linear} from "../nn/layers/linear.ts";
import {ReLu} from "../nn/layers/relu.ts";
import NetworkRenderer from "./NetworkRenderer.vue";

interface INetworkBuilder {
  list: NetworkLayer[];
  layer: NetworkLayer;
  id: number;
  network: {
    module?: LinearModule;
  }
}

// @ts-ignore
export default defineComponent<INetworkBuilder>({
  methods: {
    getLayerOptions,
    removeLayer(id: number) {
      this.list = this.list.filter((item: NetworkLayer) => item.id !== id);
    },
    addLayer() {
      this.list.push(new NetworkLayer(this.id++, this.layer.type, this.layer.inputs, this.layer.outputs));
      this.layer.type = LayerType.ReLu;
      this.inputs = 0;
      this.outputs = 0;
    },
    runNetwork() {
      const layers = this.list.map(v => v.toLayer())
      this.network.module = new LinearModule(...layers);
      console.log(this.network.module);
    }
  },
  components: {
    NetworkRenderer,
    BRow,
    BCol,
    BButton,
    BFormInput,
    BCard,
    BListGroup,
    BListGroupItem,
    draggable: VueDraggableNext,
  },
  data() {
    return {
      id: 0,
      layerOptions: getLayerOptions(),
      layer: {
        type: LayerType.ReLu,
        inputs: 0,
        outputs: 0,
      },
      network: {
        module: undefined,
      },
      list: [
        new NetworkLayer(0, LayerType.ReLu, 10, 64),
      ],
    }
  },
})
</script>

<style scoped></style>