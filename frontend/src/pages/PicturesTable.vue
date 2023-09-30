<template>
  <div>
    <el-table
        :data="pictureTableData"
        style="width: 100%"
        :header-cell-style="{'text-align':'center'}"
        :cell-style="{'text-align':'center'}"
        class="hover-show"
        @row-click="handleRowClick">
<!--        @row-click="handleFolderRowClick">-->
      <el-table-column
          prop="picture_name"
          label="Picture Name"
          min-width="40%">
        <template slot-scope="scope">
          <i class="el-icon-picture"></i>
          <span style="margin-left: 10px">{{ scope.row.pictureName }}</span>
        </template>
      </el-table-column>
      <el-table-column
          prop="picture_label"
          label="Label"
          min-width="30%">
        <template slot-scope="scope">
          <i class="el-icon-s-flag" v-if="scope.row.pictureLabel !== ''"></i>
          <span style="margin-left: 10px">{{ scope.row.pictureLabel }}</span>
        </template>
      </el-table-column>

      <el-table-column
          min-width="30%"
      >
        <template slot-scope="scope">
          <div class="hover-content" v-show="scope.row">

            <i class="el-icon-edit" style="font-size: 20px"></i>

            <i class="el-icon-delete" style="font-size: 20px"></i>

            <i v-if="!scope.row.pictureLike" class="el-icon-star-off" style="font-size: 20px"></i>
            <i v-else class="el-icon-star-on" style="font-size: 20px"></i>
          </div>
        </template>

      </el-table-column>
    </el-table>
    
    <!--   display image -->
    <el-dialog :visible.sync="dialogVisible" width="50%">
      <img v-if="selectedImage" :src="selectedImage.pictureFilePath" :alt="selectedImage.pictureName">
    </el-dialog>
    
    <div style="text-align: center">
      <el-pagination
          background
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
          :current-page.sync="pageParam.pageNo"
          layout="total, sizes, prev, pager, next, jumper"
          :page-sizes="[5, 10, 20]"
          :page-size="pageParam.pageSize"
          :total="total">
      </el-pagination>
    </div>

  </div>

</template>

<script>

import {getMyPictures} from "@/api/pictures";

export default {
  name:'PicturesTable',
  data(){
    return{
      pictureTableData:[],
      selectedImage: null,
      dialogVisible: false,
      total:0,
      pageParam:{
        pageNo:1,
        pageSize:5
      }
    }
  },
  methods:{
    showAllPictures(){
      this.pictureTableData=[]
      getMyPictures(this.pageParam).then(res=>{
        this.pictureTableData=res.data.records
        this.total=res.data.total
      })
    },
    handleSizeChange(val){
      this.pageParam.pageSize=val;
      this.showAllPictures()
    },
    handleCurrentChange(val){
      this.pageNo=val;
      this.showAllPictures()
    },
    handleRowClick(row){
      this.selectedImage = row
      this.dialogVisible = !this.dialogVisible
    }
  },
  created() {
    this.showAllPictures()
  },
  mounted() {
    this.$bus.$on('update table', (data)=>{
      this.showAllPictures()
    })
  }
}
</script>

<style scoped>

/* 在表格行悬停时显示内容 */
.hover-show .el-table__row:hover .hover-content {
  display: inline-block;
}

.hover-content {
  display: none;
}

el-pagination {
  margin-left: auto;
  margin-right: auto;
}

img {
  max-width: 100%;
  max-height: 400px;
  display: block;
  margin: 0 auto;
}

</style>