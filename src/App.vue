<script setup lang="ts">
import NetworkBuilder from "./components/NetworkBuilder.vue";
import {BCol, BContainer, BRow} from "bootstrap-vue-next";
</script>

<template>
  <BContainer fluid class="p-4">
    <BRow cols="1" class="my-3">
      <BCol>
        <NetworkBuilder/>
      </BCol>
    </BRow>
  </BContainer>
</template>

<style scoped>

</style>
