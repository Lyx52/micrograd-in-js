<script lang="ts">
import {defineComponent} from 'vue'
import {NetworkLayer} from "../render/NetworkLayer.ts";
import {Network} from "vis-network";
import {DataSet} from "vis-data";
import {v4 as uuid} from "uuid";
export default defineComponent({
  name: "NetworkRenderer",
  props: {
    layers: Array<NetworkLayer>
  },
  methods: {
    renderGraph() {
      const nodes = [];
      const edges = [];
      let id = 0;
      const HORIZONTAL_SPACING = 200;
      const VERTICAL_SPACING = 80;
      const inputNode = {
        id: 'input',
        label: 'Inputs',
        x: 0,
        y: 0,
      }

      const createLayer = (layerIndex, layer) => {
        const layerNodes = [];
        const count = layer.outputs;
        const x = (layerIndex + 1) * HORIZONTAL_SPACING;

        for (let i = 0; i < count; i++) {
          const y = (i - (count - 1) / 2) * VERTICAL_SPACING;

          layerNodes.push({
            id: id++,
            label: 'Neuron',
            x: x,
            y: y
          });
        }

        return layerNodes;
      }

      const connectNodes = (inputs, outputs) => {
        const edges = [];
        for (let i = 0; i < outputs.length; i++) {
          for (let j = 0; j < inputs.length; j++) {
            const from = inputs[j];
            const to = outputs[i];
            edges.push({
              from: from.id,
              to: to.id,
              smooth: { type: 'cubicBezier' }
            })
          }
        }

        return edges;
      }

      let previousNodes = [inputNode];
      nodes.push(inputNode)
      for (let i = 0; i < this.layers.length; i++) {
        const layer = this.layers[i];

        const currentNodes = createLayer(i, layer);

        nodes.push(...currentNodes);
        edges.push(...connectNodes(previousNodes, currentNodes));

        previousNodes = currentNodes;
      }

      const data = {
        nodes: new DataSet(nodes),
        edges: new DataSet(edges),
      };

      const renderContainer = document.querySelector('#renderContainer');
      renderContainer.style.height = '40vh';
      const network = new Network(renderContainer, data, {
        interaction: { dragNodes: false },
        physics: {
          enabled: false,
        },
      });
    }
  },
  watch: {
    layers: {
      handler(newLayers) {
        this.renderGraph(newLayers);
      },
      deep: true
    }
  }
})
</script>

<template>
  <div class="d-flex flex-column">
    <div id="renderContainer"></div>
  </div>
</template>

<style scoped>

</style>